# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded RAM-only capture of device-wide and attributed request transitions."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

from native_io_queue_results import (
    EVENT,
    UNKNOWN_ENCODING,
    UNKNOWN_STACK_CAPACITY,
    reconstruct,
    unknown_details,
)

from io_utils import fsutil

ANCHOR_MAGIC = 0x44475141
DIAGNOSTICS = ('probe_read_failure', 'missing_read_write_bio', 'request_map_full',
               'invalid_reissue', 'completion_error_or_partial_merge',
               'request_delete_failure', 'perf_submit_failure', 'sequence_map_missing',
               'invalid_flush_token', 'flush_token_delete_failure')


def load_json(path: str) -> dict:
    return json.loads(fsutil.read_text(path))


def write_json(path: str, value: dict) -> None:
    temporary = f'{path}.tmp-{os.getpid()}'
    fsutil.write_text(temporary, json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)  # Local handshakes require atomic publication.


def source_hash(path: str) -> str:
    return hashlib.sha256(fsutil.read_binary(path)).hexdigest()


def candidate_helpers(root: str) -> ModuleType:
    path = str(Path(root).resolve() / 'packages/kumo-engine/benchmark/native_io_trace.py')
    spec = importlib.util.spec_from_file_location('published_queue_helpers', path)
    if spec is None or spec.loader is None:
        raise ValueError('Candidate collector helper is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bio_next_offset(types: list[dict]) -> int:
    bios = [value for value in types if value.get('kind') == 'STRUCT' and value.get('name') == 'bio']
    members = [member for value in bios for member in value['members'] if member['name'] == 'bi_next']
    if len(members) != 1 or members[0]['bits_offset'] % 8 or members[0].get('bitfield_size', 0):
        raise ValueError('BTF BIO-chain layout is unavailable')
    by_id = {value['id']: value for value in types}
    value = by_id[members[0]['type_id']]
    while value['kind'] in {'TYPEDEF', 'CONST', 'VOLATILE', 'RESTRICT', 'TYPE_TAG'}:
        value = by_id[value['type_id']]
    if value['kind'] != 'PTR':
        raise ValueError('BTF BIO-chain link must be a pointer')
    return members[0]['bits_offset'] // 8


def flush_layout(types: list[dict]) -> dict:
    """Resolve the metadata used to prove the second, logical flush callback."""
    requests = [value for value in types if value.get('kind') == 'STRUCT' and value.get('name') == 'request']
    if len(requests) != 1:
        raise ValueError('BTF request layout is unavailable')
    by_id = {value['id']: value for value in types}
    offsets = {}
    for name, width in (('cmd_flags', 4), ('rq_flags', 4), ('__data_len', 4), ('start_time_ns', 8)):
        members = [value for value in requests[0]['members'] if value['name'] == name]
        if len(members) != 1 or members[0]['bits_offset'] % 8 or members[0].get('bitfield_size', 0):
            raise ValueError(f'Unsupported BTF request.{name} layout')
        entry = by_id[members[0]['type_id']]
        while entry['kind'] in {'TYPEDEF', 'CONST', 'VOLATILE', 'RESTRICT', 'TYPE_TAG'}:
            entry = by_id[entry['type_id']]
        if entry['kind'] != 'INT' or entry['size'] != width:
            raise ValueError(f'Unsupported BTF request.{name} width')
        offsets['OFF_REQUEST_' + name.strip('_').upper()] = members[0]['bits_offset'] // 8
    flags = [value['val'] for entry in types if entry.get('kind') in {'ENUM', 'ENUM64'}
             for value in entry.get('values', []) if value['name'] == '__RQF_FLUSH_SEQ']
    if flags != [1]:
        raise ValueError('Unsupported BTF RQF_FLUSH_SEQ enum layout')
    offsets['RQF_FLUSH_SEQ_MASK'] = 1 << flags[0]
    return offsets


def unknown_provenance(bpf, raw: bytes) -> dict:
    """Harvest only referenced UNKNOWN stacks before the BPF maps disappear."""
    details = unknown_details(raw, UNKNOWN_ENCODING)
    ids = {event['kernel_stack_id'] for event in details}
    frames = {str(index): list(bpf['unknown_stacks'].walk(index)) for index in sorted(ids) if index >= 0}
    if any(not stack for stack in frames.values()):
        raise ValueError('Successful UNKNOWN stack ID has no kernel PCs')
    symbols = {str(pc): bpf.ksym(pc).decode(errors='replace')
               for pc in sorted({pc for stack in frames.values() for pc in stack})}
    return dict(encoding_version=UNKNOWN_ENCODING, stack_capacity=UNKNOWN_STACK_CAPACITY,
                events=details, stacks=frames, symbols=symbols,
                stack_errors=sorted(index for index in ids if index < 0))


class EventBuffer:
    """Byte capacity is explicit; no event data is written before STOP/drain."""

    def __init__(self, capacity: int):
        if capacity < EVENT.size:
            raise ValueError('Event capacity is smaller than one record')
        self.capacity = capacity
        self.data = bytearray()
        self.diagnostics = dict(perf_lost=0, ram_overflow=0, record_size=0)
        self.received_sizes = {}
        self.invalid_sample = None

    def receive(self, cpu: int, pointer: int, size: int) -> None:
        self.received_sizes[size] = self.received_sizes.get(size, 0) + 1
        # Linux perf_sample_save_raw_data aligns payload + u32 size to u64.
        # BCC passes that padded RAW size; perf_output_sample skips the four
        # trailing bytes without initializing them. They are not event data.
        valid_size = size in (EVENT.size, EVENT.size + 4)
        if not valid_size:
            self.diagnostics['record_size'] += 1
            if self.invalid_sample is None:
                self.invalid_sample = ctypes.string_at(pointer, min(size, 256)).hex()
        elif len(self.data) + EVENT.size > self.capacity:
            self.diagnostics['ram_overflow'] += 1
        else:
            self.data.extend(ctypes.string_at(pointer, EVENT.size))

    def lost(self, cpu: int, count: int) -> None:
        self.diagnostics['perf_lost'] += count


def start_capture(bpf, buffer: EventBuffer) -> None:
    # BPF construction autoattaches probes. Maps start disabled, so device
    # events cannot hit an unregistered perf fd during observer setup.
    bpf['events'].open_perf_buffer(buffer.receive, lost_cb=buffer.lost, page_cnt=64)
    state = bpf['state']
    key = state.Key(0)
    value = state[key]
    if value.capture_enabled or value.pending or value.pending_flush or value.bootstrap_ns or value.end_ns:
        raise ValueError('Capture must start from a fresh disabled BPF state')
    value.capture_enabled = 1
    state[key] = value


def kernel_snapshot(bpf) -> dict:
    counters = bpf['diagnostics']
    sequence = bpf['sequence']
    state = bpf['state'][bpf['state'].Key(0)]
    return dict(diagnostics={name: sum(int(value) for value in counters[counters.Key(index)])
                             for index, name in enumerate(DIAGNOSTICS)},
                kernel_records=sum(int(value) for value in sequence[sequence.Key(0)]),
                pending=int(state.pending), bootstrap_ns=int(state.bootstrap_ns), end_ns=int(state.end_ns),
                request_entries=len(bpf['requests']), pending_flush=int(state.pending_flush),
                flush_entries=len(bpf['flush_tokens']))


def device_inflight(device: dict) -> int:
    values = [int(value) for value in fsutil.read_text(device['sysfs'] + '/inflight').split()]
    if len(values) != 2 or any(value < 0 for value in values):
        raise ValueError('Unsupported device inflight accounting')
    return sum(values)


def wait_quiet(bpf, device: dict, seconds: float, deadline: float) -> dict:
    quiet_since, previous = None, None
    while time.monotonic() < deadline:
        bpf.perf_buffer_poll(timeout=50)
        state = kernel_snapshot(bpf)
        now = time.monotonic()
        empty = (not state['pending'] and not state['request_entries'] and not state['pending_flush']
                 and not state['flush_entries'] and device_inflight(device) == 0)
        if not empty or state['kernel_records'] != previous:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= seconds:
            return state
        previous = state['kernel_records']
    raise TimeoutError('Whole physical device did not reach a verified quiet/drained state')


def collect(args) -> dict:
    if os.geteuid() != 0:
        raise PermissionError('Queue collection requires privileged BPF access')
    helpers = candidate_helpers(args.candidate_root)
    gate = load_json(args.gate)
    pid, cgroup = gate['pid'], gate['cgroup_path']
    actual = '/sys/fs/cgroup' + fsutil.read_text(f'/proc/{pid}/cgroup').strip().split('::', 1)[1]
    if cgroup != actual or not cgroup.startswith('/sys/fs/cgroup/system.slice/'):
        raise ValueError('Gate does not identify the candidate system service')
    if os.stat(cgroup).st_ino != gate['cgroup_id']:
        raise ValueError('Target cgroup identity changed')
    device = helpers.device_identity(args.device)
    backing = subprocess.check_output(['findmnt', '-T', gate['prepared_dir'], '-n', '-o', 'MAJ:MIN'], text=True).strip()
    backing_path = Path('/sys/dev/block/' + backing).resolve()
    parent = str(backing_path.parent) if fsutil.exists(str(backing_path / 'partition')) else str(backing_path)
    if fsutil.read_text(parent + '/dev').strip() != device['io_stat_key']:
        raise ValueError('Prepared inputs do not live on the measured physical device')
    btf = fsutil.read_binary('/sys/kernel/btf/vmlinux')
    types = json.loads(subprocess.check_output([
        '/usr/sbin/bpftool', '-j', 'btf', 'dump', 'file', '/sys/kernel/btf/vmlinux', 'format', 'raw']))['types']
    offsets = helpers.btf_offsets(types)
    offsets['OFF_BIO_NEXT'] = bio_next_offset(types)
    offsets.update(flush_layout(types))
    source = str(Path(__file__).with_suffix('.c'))
    constants = dict(offsets, TARGET_CGROUP=gate['cgroup_id'], DEVICE_MAJOR=device['major'],
                     DEVICE_MINOR=device['minor'], REQUEST_CAPACITY=args.request_capacity,
                     MARKER_MAGIC=helpers.MARKER_MAGIC, ANCHOR_MAGIC=ANCHOR_MAGIC, OBSERVER_PID=os.getpid())
    program = '\n'.join(f'#define {key} {value}ULL' for key, value in constants.items())
    program += '\n' + fsutil.read_text(source)
    sys.path.append(args.bcc_python_path)  # Privileged observer only; candidate environment is unchanged.
    from bcc import BPF
    buffer = EventBuffer(args.event_capacity_bytes)
    bpf = BPF(text=program)
    start_capture(bpf, buffer)
    meta = dict(gate=gate, device=device, host_kernel=os.uname().release,
                collector_sha256=source_hash(__file__), bpf_source_sha256=source_hash(source),
                btf_sha256=hashlib.sha256(btf).hexdigest(), btf_offsets=offsets,
                candidate_helper_sha256=source_hash(str(Path(helpers.__file__))),
                request_capacity=args.request_capacity, event_capacity_bytes=args.event_capacity_bytes,
                unknown_encoding_version=UNKNOWN_ENCODING,
                observer_affinity=sorted(os.sched_getaffinity(0)),
                capture_semantics='All physical-device request operations; pure target requires a complete unmixed BIO chain (inspection bound8); mixed/unchecked remain device-only. Kernel journal causality is not inferred. Bounded anonymous RAM until STOP/drain')
    record, ready, state = {}, {}, {}
    started = time.monotonic()
    try:
        # Attaching to a busy device is left-censored. Establish zero while
        # attached, before releasing the model; unknown bootstrap finals stay raw.
        deadline = time.monotonic() + args.bootstrap_seconds
        while True:
            wait_quiet(bpf, device, args.quiet_seconds, deadline)
            helpers.prepare_marker()
            helpers.marker_function()(ANCHOR_MAGIC, 0, 0, 0, 0)
            bpf.perf_buffer_poll(timeout=10)
            state = kernel_snapshot(bpf)
            if state['bootstrap_ns']:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Device became busy during bootstrap anchor')
        ready = dict(status='READY', gate_pid=pid, cgroup_id=gate['cgroup_id'], cgroup_path=cgroup,
                     bootstrap_ns=state['bootstrap_ns'], device=device,
                     collector_sha256=meta['collector_sha256'], bpf_source_sha256=meta['bpf_source_sha256'],
                     btf_sha256=meta['btf_sha256'], event_record_bytes=EVENT.size,
                     event_capacity_bytes=args.event_capacity_bytes,
                     unknown_encoding_version=UNKNOWN_ENCODING)
        write_json(args.ready, ready)
        deadline = time.monotonic() + args.timeout
        while not fsutil.exists(args.stop):
            if time.monotonic() >= deadline:
                raise TimeoutError('No STOP before collector timeout')
            bpf.perf_buffer_poll(timeout=50)
            if any(buffer.diagnostics.values()):
                raise ValueError('Raw event transport or bounded RAM failed')
        stop = load_json(args.stop)
        if type(stop['exit_code']) is not int or stop['exit_code'] != 0:
            raise ValueError('Candidate did not exit successfully')
        deadline = time.monotonic() + args.drain_seconds
        while True:
            wait_quiet(bpf, device, args.quiet_seconds, deadline)
            helpers.marker_function()(ANCHOR_MAGIC, 1, 0, 0, 0)
            bpf.perf_buffer_poll(timeout=10)
            state = kernel_snapshot(bpf)
            if state['end_ns']:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Device became busy during final capture boundary')
        capture_end = state['end_ns']
        record['unknown_provenance'] = unknown_provenance(bpf, bytes(buffer.data))
        bpf.cleanup()
        bpf = None
        row = load_json(stop['result_path'])
        raw = bytes(buffer.data)
        diagnostics = dict(state['diagnostics'], **buffer.diagnostics)
        if (any(diagnostics.values()) or state['pending'] or state['request_entries']
                or state['pending_flush'] or state['flush_entries']):
            raise ValueError('Kernel diagnostics or drain failed')
        if state['kernel_records'] * EVENT.size != len(raw):
            raise ValueError('Kernel/raw event cardinality differs')
        derived = reconstruct(raw, gate['cgroup_id'], row['iops_phase_markers'],
                              row['native_io']['fit']['after']['pid'], capture_end)
        record.update(status='PASS', capture_end_ns=capture_end, derived=derived,
                      stop=stop, diagnostics=diagnostics)
    except BaseException as error:
        if bpf is not None:
            state = kernel_snapshot(bpf)
        record.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      diagnostics=dict(state['diagnostics'], **buffer.diagnostics))
        raise
    finally:
        if bpf is not None:
            try:
                record['unknown_provenance'] = unknown_provenance(bpf, bytes(buffer.data))
            except Exception as error:
                record['unknown_provenance_error'] = f'{type(error).__name__}: {error}'
            finally:
                bpf.cleanup()
        raw = bytes(buffer.data)
        fsutil.write_binary(args.events, raw)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(meta, schema_version=1, ready=ready,
                      final_kernel_state={key: state.get(key) for key in (
                          'pending', 'request_entries', 'pending_flush', 'flush_entries', 'end_ns')},
                      transport_sizes=buffer.received_sizes, invalid_transport_sample=buffer.invalid_sample,
                      events=dict(path=args.events, sha256=hashlib.sha256(raw).hexdigest(),
                                  bytes=len(raw), records=len(raw) // EVENT.size,
                                  kernel_records=state.get('kernel_records', 0), format=EVENT.format),
                      observer_user_seconds=usage.ru_utime, observer_system_seconds=usage.ru_stime,
                      observer_peak_rss_kib=usage.ru_maxrss, observation_seconds=time.monotonic() - started,
                      observer_cpu_scope='Whole observer including setup/compilation/reconstruction; not model wall attribution')
        write_json(args.output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('collect')
    for name in ('gate', 'ready', 'stop', 'output', 'events', 'candidate-root'):
        command.add_argument('--' + name, required=True)
    command.add_argument('--device', default='/dev/nvme0n1')
    command.add_argument('--request-capacity', type=int, default=65536)
    command.add_argument('--event-capacity-bytes', type=int, default=536870912)
    command.add_argument('--timeout', type=float, default=3600)
    command.add_argument('--bootstrap-seconds', type=float, default=30)
    command.add_argument('--drain-seconds', type=float, default=10)
    command.add_argument('--quiet-seconds', type=float, default=1)
    command.add_argument('--bcc-python-path', default='/usr/lib/python3/dist-packages')
    args = parser.parse_args()
    if (args.request_capacity < 1 or args.event_capacity_bytes < EVENT.size
            or any(not math.isfinite(value) or value <= 0 for value in (
                args.timeout, args.bootstrap_seconds, args.drain_seconds, args.quiet_seconds))
            or args.quiet_seconds >= min(args.drain_seconds, args.bootstrap_seconds)):
        parser.error('Positive bounded capacity/timeouts and quiet<drain/bootstrap are required')
    collect(args)


if __name__ == '__main__':
    main()
