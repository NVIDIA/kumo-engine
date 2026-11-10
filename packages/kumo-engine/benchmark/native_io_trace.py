# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only attributed block-request tracing and synchronous phase marks."""

import argparse
import ctypes
import errno
import hashlib
import json
import os
import resource
import stat
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

from io_utils import fsutil

MARKER_MAGIC = 0x4447494F
PHASES = {0: 'total', 1: 'fit', 2: 'training', 3: 'predict',
          4: 'repeat', 5: 'reload', 6: 'e2e'}
MARKERS = [(6, 1), (1, 1), (2, 1), (2, 0), (1, 0), (3, 1), (3, 0), (6, 0),
           (4, 1), (4, 0), (5, 1), (5, 0)]


@cache
def marker_function():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_ulong, ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    return libc.prctl


def prepare_marker() -> None:
    """Resolve ctypes before timed phases, equally for tracing on and off."""
    marker_function()


def mark_phase(phase_id: int, entering: bool) -> dict[str, int]:
    """Mark an interval without changing kernel process state.

    The collector observes the intentionally unsupported prctl option before
    Linux returns EINVAL. A missing collector cannot masquerade as successful
    measurement: its final output must independently match all twelve marks.
    """
    if phase_id not in range(1, 7) or not isinstance(entering, bool):
        raise ValueError('Expected phase1..6 and a boolean transition')
    prctl = marker_function()
    before = time.perf_counter_ns()
    result = prctl(MARKER_MAGIC, phase_id, int(entering), 0, 0)
    after = time.perf_counter_ns()
    error = ctypes.get_errno()
    if result != -1 or error != errno.EINVAL:
        raise RuntimeError(f'Unsupported marker prctl returned {result}/{error}')
    return dict(pid=os.getpid(), phase_id=phase_id, entering=int(entering),
                before_ns=before, after_ns=after)


def load_json(path: str) -> dict:
    return json.loads(fsutil.read_text(path))


def save_json(path: str, value: dict) -> None:
    temporary = f'{path}.tmp-{os.getpid()}-{time.monotonic_ns()}'
    fsutil.write_text(temporary, json.dumps(value, indent=2, allow_nan=False) + '\n')
    # fsutil.atomic_write_text currently has no local backend registration.
    # A local rename is required for the service/observer readiness handshake.
    os.replace(temporary, path)


def btf_offsets(types: list[dict]) -> dict[str, int]:
    """Resolve private kernel layouts from actual BTF, never guessed offsets."""
    by_id = {item['id']: item for item in types}
    fields = {
        'request': {'q': 8, 'cmd_flags': 4, '__data_len': 4, 'bio': 8},
        'request_queue': {'disk': 8}, 'gendisk': {'major': 4, 'first_minor': 4},
        'bio': {'bi_blkg': 8, 'bi_opf': 4}, 'blkcg_gq': {'blkcg': 8},
        'blkcg': {'css': None}, 'cgroup_subsys_state': {'cgroup': 8},
        'cgroup': {'kn': 8}, 'kernfs_node': {'id': 8},
    }
    offsets = {}
    for name, required in fields.items():
        matches = [t for t in types if t.get('kind') == 'STRUCT' and t.get('name') == name]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous/missing BTF struct {name}')
        members = {m['name']: m for m in matches[0]['members']}
        for member, width in required.items():
            value = members[member]
            if value['bits_offset'] % 8 or value.get('bitfield_size', 0):
                raise ValueError(f'Unsupported BTF bit field {name}.{member}')
            entry = by_id[value['type_id']]
            while entry['kind'] in {'TYPEDEF', 'CONST', 'VOLATILE', 'RESTRICT', 'TYPE_TAG'}:
                entry = by_id[entry['type_id']]
            size = 8 if entry['kind'] == 'PTR' else entry.get('size')
            if width is not None and size != width:
                raise ValueError(f'Unexpected BTF width {name}.{member}: {size}')
            key = 'OFF_' + name.upper() + '_' + member.strip('_').upper()
            offsets[key] = value['bits_offset'] // 8
    enum_values = {v['name']: v['val'] for item in types
                   if item.get('kind') in {'ENUM', 'ENUM64'}
                   for v in item.get('values', [])}
    if enum_values.get('REQ_OP_READ') != 0 or enum_values.get('REQ_OP_WRITE') != 1:
        raise ValueError('Unsupported request operation layout')
    offsets['ROOT_ISSUE_FLAGS'] = ((1 << enum_values['__REQ_META'])
                                   | (1 << enum_values['__REQ_SWAP']))
    return offsets


def device_identity(path: str) -> dict:
    metadata = os.stat(path)  # Linux device metadata has no fsutil equivalent.
    if not stat.S_ISBLK(metadata.st_mode):
        raise ValueError('Tracing requires a physical block-device path')
    major, minor = os.major(metadata.st_rdev), os.minor(metadata.st_rdev)
    sysfs = str(Path(f'/sys/dev/block/{major}:{minor}').resolve())
    if fsutil.exists(sysfs + '/partition'):
        raise ValueError('Use the parent physical device, not its partition')
    if not fsutil.exists(sysfs + '/device'):
        raise ValueError('Stacked/virtual block devices require explicit topology support')
    return dict(path=path, major=major, minor=minor, sysfs=sysfs,
                io_stat_key=f'{major}:{minor}')


def validate_trace(record: dict, client_markers: list[dict], model_pid: int) -> dict:
    """Fail closed on incomplete accounting, protocol errors or clock mismatch."""
    if record['pending_reads'] or record['pending_writes'] or record['request_entries']:
        raise ValueError('Undrained request accounting')
    if any(record['diagnostics'][name] for name in FATAL_DIAGNOSTICS):
        raise ValueError(f"Collector diagnostics: {record['diagnostics']}")
    for count in record['counts'].values():
        if any(type(value) is not int or value < 0 for value in count.values()):
            raise ValueError('Invalid completion counter')
        if count['carried_in_reads'] > count['reads']:
            raise ValueError('Carried-in completions exceed phase completions')
    total = record['counts']['total']
    if total['issued'] != total['reads'] + total['writes'] + total['merged_away']:
        raise ValueError('Issued/final/merged accounting does not reconcile')
    counts = record['counts']
    for field in ('reads', 'read_bytes', 'writes', 'write_bytes'):
        if any(count[field] > total[field] for count in counts.values()):
            raise ValueError('Phase counter exceeds total')
        if (counts['training'][field] > counts['fit'][field]
                or counts['fit'][field] + counts['predict'][field] > counts['e2e'][field]
                or counts['e2e'][field] + counts['repeat'][field] + counts['reload'][field] > total[field]):
            raise ValueError('Nested/disjoint phase accounting is impossible')
    kernel_marks = record['markers']
    if len(kernel_marks) != len(MARKERS) or len(client_markers) != len(MARKERS):
        raise ValueError('Expected twelve phase marks')
    intervals, active, starts = {}, 0, {}
    for expected, kernel, client in zip(MARKERS, kernel_marks, client_markers, strict=True):
        phase, entering = expected
        if (kernel['phase_id'], kernel['entering']) != expected:
            raise ValueError('Unexpected kernel phase order')
        if (client['phase_id'], client['entering']) != expected:
            raise ValueError('Unexpected client phase order')
        if kernel['pid'] != model_pid or client['pid'] != model_pid:
            raise ValueError('Phase PID does not match the model process')
        if not client['before_ns'] <= kernel['before_ns'] <= kernel['after_ns'] <= client['after_ns']:
            raise ValueError('BPF CLOCK_MONOTONIC is outside perf_counter_ns bracket')
        bit = 1 << phase
        active = (active | bit) if entering else (active & ~bit)
        if kernel['mask'] != active:
            raise ValueError('Phase mask transition mismatch')
        if entering:
            starts[phase] = kernel
        else:
            first = starts[phase]
            lower = kernel['before_ns'] - first['after_ns']
            upper = kernel['after_ns'] - first['before_ns']
            elapsed = kernel['after_ns'] - first['after_ns']
            if lower <= 0:
                raise ValueError('Nonpositive phase duration')
            count = record['counts'][PHASES[phase]]
            intervals[PHASES[phase]] = dict(
                seconds=elapsed / 1e9, lower_seconds=lower / 1e9,
                upper_seconds=upper / 1e9, boundary_uncertainty_ns=upper - lower,
                completed_read_requests=count['reads'],
                completed_read_requests_per_second=count['reads'] * 1e9 / elapsed,
                completed_read_bytes=count['read_bytes'],
                carried_in_reads=count['carried_in_reads'],
                carried_out_reads=count['carried_out_reads'],
                pending_reads_at_start=first['pending_reads'],
                pending_reads_at_end=kernel['pending_reads'])
    return intervals


def validate_collection(trace: dict, result: dict, gate: dict) -> None:
    """Replay eligibility from saved fields instead of trusting status alone."""
    if trace['status'] != 'PASS' or trace['gate'] != gate:
        raise ValueError('Failed collector or different gate identity')
    ready = trace['ready']
    if ready['status'] != 'READY' or ready['cgroup_id'] != gate['cgroup_id']:
        raise ValueError('Collector READY identity mismatch')
    for key in ('cgroup_id', 'device', 'collector_sha256', 'bpf_source_sha256',
                'request_capacity', 'marker_capacity', 'btf_sha256'):
        if trace[key] != ready[key]:
            raise ValueError(f'Collector identity changed: {key}')
    device = trace['device']
    if device['io_stat_key'] != f"{device['major']}:{device['minor']}":
        raise ValueError('Device accounting identity mismatch')
    phases = validate_trace(trace, result['iops_phase_markers'],
                            result['native_io']['fit']['after']['pid'])
    if trace['phases'] != phases:
        raise ValueError('Saved phase rates do not match raw counters/markers')


FATAL_DIAGNOSTICS = ('probe_read_failure', 'missing_bio_origin', 'request_map_full',
                     'unexpected_reissue', 'request_error', 'marker_protocol',
                     'marker_capacity', 'partial_merge_unsupported', 'marker_pid_changed',
                     'request_delete_failure')
DIAGNOSTICS = FATAL_DIAGNOSTICS + ('untracked_device_completions',)


def count_fields(value) -> dict[str, int]:
    return dict(reads=int(value.reads), read_bytes=int(value.read_bytes),
                writes=int(value.writes), write_bytes=int(value.write_bytes),
                carried_in_reads=int(value.carried_in_reads),
                carried_out_reads=int(value.carried_out_reads),
                root_issued_reads=int(value.root_issued_reads),
                root_issued_writes=int(value.root_issued_writes),
                issued=int(value.issued), merged_away=int(value.merged_away),
                requeues=int(value.requeues), reissues=int(value.reissues),
                partials=int(value.partials))


def snapshot(bpf) -> dict:
    counts = bpf['counts']
    summed = {}
    for phase, name in PHASES.items():
        values = [count_fields(value) for value in counts[counts.Key(phase)]]
        summed[name] = {key: sum(v[key] for v in values) for key in values[0]}
    diagnostics = bpf['diagnostics']
    errors = {name: sum(int(v) for v in diagnostics[diagnostics.Key(index)])
              for index, name in enumerate(DIAGNOSTICS)}
    state = bpf['state'][bpf['state'].Key(0)]
    markers = bpf['markers']
    marks = []
    for index in range(min(int(state.marker_count), 16)):
        mark = markers[markers.Key(index)]
        marks.append(dict(before_ns=int(mark.before_ns), after_ns=int(mark.after_ns),
                          pending_reads=int(mark.pending_reads), pending_writes=int(mark.pending_writes),
                          pid=int(mark.pid), phase_id=int(mark.phase), entering=int(mark.entering),
                          mask=int(mark.mask)))
    return dict(counts=summed, diagnostics=errors, markers=marks,
                pending_reads=int(state.pending_reads), pending_writes=int(state.pending_writes),
                request_entries=len(bpf['requests']))


def collect(args) -> dict:
    if os.geteuid() != 0:
        raise PermissionError('Collector requires temporary privileged BPF access')
    gate = load_json(args.gate)
    pid, cgroup = gate['pid'], gate['cgroup_path']
    expected = '/sys/fs/cgroup' + fsutil.read_text(f'/proc/{pid}/cgroup').strip().split('::', 1)[1]
    if cgroup != expected or not cgroup.startswith('/sys/fs/cgroup/system.slice/'):
        raise ValueError('Gate must name the existing target system-service cgroup')
    cgroup_id = os.stat(cgroup).st_ino  # Kernel cgroup identity, not file data.
    if gate['cgroup_id'] != cgroup_id:
        raise ValueError('Gate cgroup identity changed')
    if not fsutil.exists(cgroup + '/io.stat'):
        raise ValueError('Target cgroup has no delegated I/O controller')
    device = device_identity(args.device)
    backing = subprocess.check_output(['findmnt', '-T', gate['prepared_dir'],
                                       '-n', '-o', 'MAJ:MIN'], text=True).strip()
    backing_sysfs = Path('/sys/dev/block/' + backing).resolve()
    parent = str(backing_sysfs.parent) if fsutil.exists(str(backing_sysfs / 'partition')) else str(backing_sysfs)
    physical = fsutil.read_text(parent + '/dev').strip()
    if physical != device['io_stat_key']:
        raise ValueError('Requested device is not the prepared-input physical device')
    device.update(prepared_path=gate['prepared_dir'], filesystem_device=backing)
    btf_raw = subprocess.check_output(['/usr/sbin/bpftool', '-j', 'btf', 'dump',
                                      'file', '/sys/kernel/btf/vmlinux', 'format', 'raw'])
    offsets = btf_offsets(json.loads(btf_raw)['types'])
    constants = dict(offsets, TARGET_CGROUP=cgroup_id, DEVICE_MAJOR=device['major'],
                     DEVICE_MINOR=device['minor'], REQUEST_CAPACITY=args.request_capacity,
                     MARKER_MAGIC=MARKER_MAGIC)
    source = str(Path(__file__).with_suffix('.c'))
    program = '\n'.join(f'#define {key} {value}ULL' for key, value in constants.items())
    program += '\n' + fsutil.read_text(source)
    # Installed distro BCC is pure Python/ctypes. This path is local to the
    # privileged observer and does not modify the model's dependency search.
    sys.path.append(args.bcc_python_path)
    from bcc import BPF
    bpf = BPF(text=program)
    started = time.perf_counter_ns()
    meta = dict(cgroup_id=cgroup_id, cgroup_path=cgroup, gate_pid=pid, device=device,
                host_kernel=os.uname().release, btf_offsets=offsets,
                btf_sha256=hashlib.sha256(fsutil.read_binary('/sys/kernel/btf/vmlinux')).hexdigest(),
                collector_sha256=hashlib.sha256(fsutil.read_binary(__file__)).hexdigest(),
                bpf_source_sha256=hashlib.sha256(fsutil.read_binary(source)).hexdigest(),
                request_capacity=args.request_capacity, marker_capacity=16,
                phase_counter_capacity=7, per_completion_userspace_events=0,
                count_semantics='Target BIO-origin final device-request callbacks during active phase; finite boundary uncertainty',
                byte_semantics='Completed request payload assigned to final-callback phase; not bytes transferred exclusively inside that window',
                attribution_scope='Exact target BIO-origin cgroup; metadata/swap root-issue flag counted separately; not SSD internal operations')
    record = {}
    try:
        ready = dict(status='READY', monotonic_ns=started, **meta)
        save_json(args.ready, ready)
        deadline = time.monotonic() + args.timeout
        while not fsutil.exists(args.stop):
            if time.monotonic() >= deadline:
                raise TimeoutError('No stop record before collector timeout')
            time.sleep(0.1)
        stop = load_json(args.stop)
        if stop['exit_code'] != 0:
            raise ValueError('Target failed; collector cannot qualify the trial')
        deadline, quiet_since, previous = time.monotonic() + args.drain_seconds, None, None
        while True:
            record = snapshot(bpf)
            total = record['counts']['total']
            current = (total['issued'], total['reads'], total['writes'], total['merged_away'])
            empty = not (record['pending_reads'] or record['pending_writes'] or record['request_entries'])
            now = time.monotonic()
            quiet_since = quiet_since if empty and current == previous else now
            if empty and now - quiet_since >= args.quiet_seconds:
                break
            if now >= deadline:
                raise TimeoutError('Target requests failed to drain and become quiet')
            previous = current
            time.sleep(0.1)
        bpf.cleanup()
        bpf = None
        result = load_json(stop['result_path'])
        intervals = validate_trace(record, result['iops_phase_markers'],
                                   result['native_io']['fit']['after']['pid'])
        record.update(status='PASS', phases=intervals)
    except BaseException as error:
        if bpf is not None:
            record = snapshot(bpf)
        record.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if bpf is not None:
            bpf.cleanup()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(meta, gate=gate, ready=ready, observer_user_seconds=usage.ru_utime,
                      observer_system_seconds=usage.ru_stime,
                      observer_peak_rss_kib=usage.ru_maxrss,
                      observer_cpu_scope='Whole observer process including imports/BCC compile/setup, not only observation_seconds',
                      observation_seconds=(time.perf_counter_ns() - started) / 1e9)
        save_json(args.output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('collect')
    for name in ('gate', 'ready', 'stop', 'output'):
        command.add_argument('--' + name, required=True)
    command.add_argument('--device', default='/dev/nvme0n1')
    command.add_argument('--request-capacity', type=int, default=8192)
    command.add_argument('--drain-seconds', type=float, default=10)
    command.add_argument('--quiet-seconds', type=float, default=1)
    command.add_argument('--timeout', type=float, default=3600)
    command.add_argument('--bcc-python-path', default='/usr/lib/python3/dist-packages')
    args = parser.parse_args()
    if args.request_capacity < 1 or not 0 < args.quiet_seconds < args.drain_seconds:
        parser.error('Positive capacity and0<quiet<drain are required')
    collect(args)


if __name__ == '__main__':
    main()
