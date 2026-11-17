# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded cgroup CPU samples or classified off-CPU intervals, never both."""

import argparse
import ctypes
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from io_utils import fsutil

MARKERS = [(6, 1), (1, 1), (2, 1), (2, 0), (1, 0), (3, 1), (3, 0),
           (6, 0), (4, 1), (4, 0), (5, 1), (5, 0)]
PHASES = {'fit': 1, 'training': 2, 'predict': 3, 'repeat': 4, 'reload': 5, 'e2e': 6}
INTERVAL = struct.Struct('<QQQIIIIii16s')
PROGRAM = r'''
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
_Static_assert(TASK_REPORT_MAX == 256, "Review sched_switch state classification on this kernel");
struct key_t { u32 tgid; u32 tid; s32 user; s32 kernel; u32 mask; char comm[16]; };
struct wait_t { u64 started; u64 state; struct key_t key; };
struct interval_t {
    u64 started; u64 ended; u64 state;
    u32 tgid; u32 tid; u32 mask; u32 reserved;
    s32 user; s32 kernel; char comm[16];
};
struct marker_t { u64 ns; u32 pid; u32 tid; u32 phase; u32 entering; u32 mask; };
BPF_ARRAY(phase_mask, u32, 1);
BPF_ARRAY(enabled, u32, 1);
BPF_ARRAY(markers, struct marker_t, 12);
BPF_ARRAY(marker_count, u32, 1);
BPF_ARRAY(failures, u64, 3);
BPF_HASH(samples, struct key_t, u64, 65536);
BPF_HASH(waits, u32, struct wait_t, 65536);
BPF_STACK_TRACE(stacks, 65536);
BPF_PERF_OUTPUT(intervals);
static void failure(u32 index) { u64 *v = failures.lookup(&index); if (v) __sync_fetch_and_add(v, 1); }
static int active(void) { u32 zero = 0; u32 *value = enabled.lookup(&zero); return value && *value; }
static int targeted(void) { return active() && bpf_get_current_cgroup_id() == TARGET_CGROUP; }
static void identity(struct key_t *key) {
    u64 id = bpf_get_current_pid_tgid(); key->tgid = id >> 32; key->tid = id;
    u32 zero = 0; u32 *mask = phase_mask.lookup(&zero); key->mask = mask ? *mask : 0;
    bpf_get_current_comm(key->comm, sizeof(key->comm));
}
TRACEPOINT_PROBE(syscalls, sys_enter_prctl) {
    if (!targeted() || args->option != 0x4447494f) return 0;
    u64 identity = bpf_get_current_pid_tgid();
    if ((u32)identity != (u32)(identity >> 32)) { failure(0); return 0; }
    u32 zero = 0, phase = args->arg2, entering = args->arg3;
    u32 *count = marker_count.lookup(&zero), *mask = phase_mask.lookup(&zero);
    if (!count || !mask || *count >= 12 || phase < 1 || phase > 6 || entering > 1) {
        failure(0); return 0;
    }
    if (entering) *mask |= 1 << phase; else *mask &= ~(1 << phase);
    struct marker_t marker = {.ns=bpf_ktime_get_ns(), .pid=identity >> 32, .tid=(u32)identity,
        .phase=phase, .entering=entering, .mask=*mask};
    // The twelve accepted markers come from one model thread; replay checks
    // their exact order and PID. Avoid unsupported return-value XADD on ARM.
    u32 index = *count; *count = index + 1; markers.update(&index, &marker);
    return 0;
}
int cpu_sample(struct bpf_perf_event_data *ctx) {
    if (!targeted()) return 0;
    struct key_t key = {}; identity(&key);
    key.user = stacks.get_stackid(ctx, BPF_F_USER_STACK);
    key.kernel = stacks.get_stackid(ctx, 0);
    u64 initial = 0, *value = samples.lookup_or_try_init(&key, &initial);
    if (value) __sync_fetch_and_add(value, 1); else failure(1);
    return 0;
}
TRACEPOINT_PROBE(sched, sched_switch) {
    if (!active()) return 0;
    u64 now = bpf_ktime_get_ns(); u32 next = args->next_pid;
    struct wait_t *prior = waits.lookup(&next);
    if (prior) {
        struct interval_t record = {.started=prior->started, .ended=now, .state=prior->state,
            .tgid=prior->key.tgid, .tid=next, .mask=prior->key.mask,
            .user=prior->key.user, .kernel=prior->key.kernel};
        __builtin_memcpy(record.comm, prior->key.comm, sizeof(record.comm));
        if (intervals.perf_submit(args, &record, sizeof(record))) failure(1);
        if (waits.delete(&next)) failure(2);
    }
    if (!targeted()) return 0;
    if (args->prev_state & (EXIT_DEAD | EXIT_ZOMBIE)) return 0;
    struct wait_t value = {.started=now, .state=args->prev_state}; identity(&value.key);
    value.key.user = stacks.get_stackid(args, BPF_F_USER_STACK);
    value.key.kernel = stacks.get_stackid(args, 0);
    u32 prev = args->prev_pid;
    if (waits.update(&prev, &value)) failure(1);
    return 0;
}
'''


def load_json(path: str) -> dict:
    return json.loads(fsutil.read_text(path))


def save_json(path: str, value: dict) -> None:
    temporary = path + '.writing'
    fsutil.write_text(temporary, json.dumps(value, indent=2, allow_nan=False) + '\n')
    # These are local Linux process gates; fsutil has no atomic rename.
    os.replace(temporary, path)


def profile_program(cgroup_id: int, mode: str) -> str:
    if cgroup_id < 1 or mode not in {'cpu', 'offcpu'}:
        raise ValueError('Invalid profile cgroup or provider')
    program = PROGRAM.replace('TARGET_CGROUP', str(cgroup_id))
    if mode == 'cpu':
        return program[:program.index('TRACEPOINT_PROBE(sched, sched_switch)')]
    start = program.index('int cpu_sample(')
    end = program.index('TRACEPOINT_PROBE(sched, sched_switch)')
    return program[:start] + program[end:]


def phase_windows(markers: list[dict], clients: list[dict], pid: int) -> dict:
    if len(markers) != 12 or len(clients) != 12:
        raise ValueError('Expected exactly twelve phase edges')
    starts, windows = {}, {}
    prior, mask = 0, 0
    for expected, kernel, client in zip(MARKERS, markers, clients, strict=True):
        phase, entering = expected
        if (kernel['phase'], kernel['entering']) != expected:
            raise ValueError('Unexpected profile marker order')
        if (client['phase_id'], client['entering']) != expected:
            raise ValueError('Client/profile marker order mismatch')
        if kernel['pid'] != pid or kernel['tid'] != pid or client['pid'] != pid:
            raise ValueError('Profile marker PID differs from actual model PID')
        if not client['before_ns'] <= kernel['ns'] <= client['after_ns'] or kernel['ns'] <= prior:
            raise ValueError('Profile clock is outside its client syscall bracket')
        mask = mask | (1 << phase) if entering else mask & ~(1 << phase)
        if kernel['mask'] != mask:
            raise ValueError('Profile phase bitmask mismatch')
        if entering:
            starts[phase] = kernel['ns']
        else:
            windows[phase] = (starts[phase], kernel['ns'])
        prior = kernel['ns']
    return {name: windows[phase] for name, phase in PHASES.items()}


def interval_totals(events: bytes, windows: dict) -> dict:
    if len(events) % INTERVAL.size:
        raise ValueError('Truncated off-CPU interval record')
    totals = {name: {'blocked_sleep_ns': 0, 'runnable_or_preempted_ns': 0}
              for name in windows}
    for start, end, state, tgid, tid, mask, reserved, user, kernel, comm in INTERVAL.iter_unpack(events):
        if end <= start or not tgid or not tid or reserved:
            raise ValueError('Invalid off-CPU interval')
        # sched_switch uses TASK_REPORT_MAX (256) for preemption. Header
        # classification is checked by the collector instead of treating R+
        # as blocked sleep. Preserve the raw state in every interval.
        category = 'blocked_sleep_ns' if state & 255 else 'runnable_or_preempted_ns'
        for phase, (left, right) in windows.items():
            totals[phase][category] += max(0, min(end, right) - max(start, left))
    return totals


def phase_samples(samples: list[dict], frequency: int) -> dict:
    if frequency < 1:
        raise ValueError('Sampling frequency must be positive')
    counts = {phase: 0 for phase in PHASES}
    for sample in samples:
        if sample['count'] <= 0 or sample['mask'] & ~126:
            raise ValueError('Invalid CPU sample count or phase mask')
        for phase, number in PHASES.items():
            if sample['mask'] & (1 << number):
                counts[phase] += sample['count']
    return {phase: {'samples': count, 'estimated_thread_seconds': count / frequency}
            for phase, count in counts.items()}


def interval_payload(sample: bytes) -> bytes:
    if not INTERVAL.size <= len(sample) <= INTERVAL.size + 7 or any(sample[INTERVAL.size:]):
        raise ValueError('Unexpected perf sample length or nonzero alignment padding')
    return sample[:INTERVAL.size]


def unwind_coverage(samples: list[dict], events: bytes, stacks: dict,
                    symbols: dict, training: tuple[int, int], depth_cap: int = 127) -> dict:
    if samples:
        entries = [(item['tgid'], item['user'], item['kernel'], item['count'])
                   for item in samples if item['mask'] & 4]
        unit = 'training_CPU_samples'
    else:
        left, right = training
        entries = [(item[3], item[7], item[8], max(0, min(item[1], right) - max(item[0], left)))
                   for item in INTERVAL.iter_unpack(events)]
        unit = 'training_offCPU_thread_nanoseconds'
    total = usable = truncated = kernel_usable = 0
    for pid, user, kernel, weight in entries:
        total += weight
        frames = stacks.get(str(user), [])
        if any(symbols.get(f'{pid}:{pc}', {}).get('symbol', '??') != '??' for pc in frames):
            usable += weight
        if len(frames) >= depth_cap:
            truncated += weight
        if stacks.get(str(kernel)):
            kernel_usable += weight
    return {'unit': unit, 'total_weight': total, 'user_usable_weight': usable,
            'user_failed_weight': total - usable, 'user_usable_fraction': usable / max(1, total),
            'possible_truncated_weight': truncated,
            'possible_truncated_fraction': truncated / max(1, total),
            'kernel_usable_weight': kernel_usable}


def sampling_coverage(counts: dict, row: dict) -> dict:
    interval = row['work']['training_interval']
    before, after = interval['resources_before']['cpu'], interval['resources_after']['cpu']
    cpu = (after['usage_usec'] - before['usage_usec']) / 1e6
    if cpu <= 0:
        raise ValueError('CPU profile has no positive cgroup training CPU accounting')
    sampled = counts['training']['estimated_thread_seconds']
    throttled = after.get('nr_throttled', 0) - before.get('nr_throttled', 0)
    return {'cgroup_cpu_seconds': cpu, 'estimated_sampled_cpu_seconds': sampled,
            'sampled_to_accounted_ratio': sampled / cpu, 'throttled_periods': throttled,
            'scope': 'All target cgroup tasks; sampling observes running CPU time only',
            'qualification': 'THROTTLED_CONTEXT' if throttled else 'RUNNING_CPU_ONLY',
            'idle_inference': 'Unsupported: absence of samples does not identify idle or blocked time'}


def profile_snapshot(bpf, mode: str, events: bytes) -> dict:
    count = bpf['marker_count'][ctypes.c_int(0)].value
    markers = []
    for index in range(min(count, 12)):
        item = bpf['markers'][ctypes.c_int(index)]
        markers.append({'ns': int(item.ns), 'pid': int(item.pid), 'tid': int(item.tid),
                        'phase': int(item.phase),
                        'entering': int(item.entering), 'mask': int(item.mask)})
    samples = []
    if mode == 'cpu':
        for key, value in bpf['samples'].items():
            samples.append({'tgid': key.tgid, 'tid': key.tid, 'user': key.user,
                            'kernel': key.kernel, 'mask': key.mask,
                            'comm': key.comm.decode(errors='replace'), 'count': value.value})
    stack_ids = {item[name] for item in samples for name in ('user', 'kernel')}
    for item in INTERVAL.iter_unpack(events):
        stack_ids.update(item[7:9])
    return {'samples': samples, 'markers': markers,
            'stacks': {str(index): list(bpf['stacks'].walk(index))
                       for index in stack_ids if index >= 0},
            'failures': [int(bpf['failures'][ctypes.c_int(index)].value) for index in range(3)],
            'pending_at_stop': len(bpf['waits'])}


def mapped_address(pc: int, maps: str) -> tuple[str, int] | None:
    """Map a process PC to its ELF file offset using captured /proc maps."""
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        left, right = (int(value, 16) for value in fields[0].split('-'))
        if left <= pc < right:
            return fields[5], pc - left + int(fields[2], 16)
    return None


def symbolize_users(stacks: dict, users: set[tuple[int, int]], maps: dict) -> dict:
    # ELF files are read only after STOP. Resolving while the workload runs
    # would introduce observer storage traffic into whole-device measurements.
    grouped: dict[str, set[tuple[int, int, int]]] = {}
    for pid, index in sorted(users):
        for pc in stacks.get(str(index), []):
            mapped = mapped_address(pc, maps.get(str(pid), ''))
            if mapped is not None:
                path, offset = mapped
                grouped.setdefault(path, set()).add((pid, pc, offset))
    resolved = {}
    for path, addresses in grouped.items():
        if not fsutil.exists(path):
            continue
        headers = subprocess.check_output(['readelf', '-lW', path], text=True)
        loads = [(int(fields[1], 16), int(fields[4], 16), int(fields[2], 16))
                 for line in headers.splitlines() if (fields := line.split())
                 and fields[0] == 'LOAD']
        virtual = []
        selected = []
        for pid, pc, offset in sorted(addresses):
            for start, size, address in loads:
                if start <= offset < start + size:
                    virtual.append(hex(offset - start + address))
                    selected.append((pid, pc))
                    break
        if not virtual:
            continue
        result = subprocess.run(['addr2line', '-C', '-f', '-e', path],
                                input='\n'.join(virtual) + '\n', capture_output=True,
                                text=True, check=True)
        lines = result.stdout.splitlines()
        if len(lines) != 2 * len(selected):
            raise ValueError('Incomplete userspace symbolization output')
        for (pid, pc), symbol, location in zip(selected, lines[::2], lines[1::2], strict=True):
            resolved[f'{pid}:{pc}'] = {'module': path, 'symbol': symbol, 'location': location}
    return resolved


def validate_collection(record: dict, row: dict, gate: dict, events: bytes) -> None:
    if record['status'] != 'PASS' or record['gate'] != gate:
        raise ValueError('Failed or mismatched profile capture')
    ready = record['ready']
    if ready['status'] != 'READY' or type(record['stop']['exit_code']) is not int or record['stop'] != {
            'exit_code': 0, 'result_path': row['matrix_spec']['result']}:
        raise ValueError('Profile READY/STOP does not identify this successful workload')
    for key in ('gate_pid', 'cgroup_id', 'cgroup_path', 'collector_sha256', 'program_sha256',
                'mode', 'btf_sha256', 'kernel_release', 'event_capacity_bytes'):
        if record[key] != ready[key]:
            raise ValueError('Profile identity changed after READY')
    for key in ('collector_sha256', 'program_sha256', 'btf_sha256'):
        if len(record[key]) != 64 or any(char not in '0123456789abcdef' for char in record[key]):
            raise ValueError('Profile source/BTF hash is invalid')
    if (record['gate_pid'], record['cgroup_id'], record['cgroup_path']) != (
            gate['pid'], gate['cgroup_id'], gate['cgroup_path']):
        raise ValueError('Profile cgroup differs from the requested service')
    if hashlib.sha256(events).hexdigest() != record['events']['sha256']:
        raise ValueError('Profile event bytes do not match their source hash')
    if (record['events']['bytes'] != len(events) or len(events) > record['event_capacity_bytes']
            or record['events']['format'] != INTERVAL.format):
        raise ValueError('Profile event size differs from the bounded capture')
    if any(record['failures']) or record['lost_events'] or record['overflow']:
        raise ValueError('Profile lost data or exhausted a bounded map/buffer')
    program = profile_program(record['cgroup_id'], record['mode'])
    if record['program_sha256'] != hashlib.sha256(program.encode()).hexdigest():
        raise ValueError('Captured profile program differs from its requested cgroup/provider')
    windows = phase_windows(record['markers'], row['iops_phase_markers'],
                            row['native_io']['fit']['after']['pid'])
    if record['mode'] == 'offcpu':
        if record['phase_thread_time'] != interval_totals(events, windows):
            raise ValueError('Off-CPU phase totals differ from raw intervals')
        if record['pending_at_stop']:
            raise ValueError('Off-CPU intervals remained open at capture end')
    elif record['mode'] == 'cpu':
        if events or not record['samples']:
            raise ValueError('CPU sampling did not record any target thread samples')
        if record['phase_cpu_samples'] != phase_samples(record['samples'], record['frequency_hz_per_cpu']):
            raise ValueError('CPU phase totals differ from the sampled phase masks')
        coverage = sampling_coverage(record['phase_cpu_samples'], row)
        if record['sampling_coverage'] != coverage:
            raise ValueError('CPU sample coverage differs from actual cgroup CPU accounting')
        if not .7 <= coverage['sampled_to_accounted_ratio'] <= 1.3:
            raise ValueError('CPU sampling missed its expected running-time coverage')
    else:
        raise ValueError('Unknown profiling mode')
    if not record['stack_quality']['kernel_nonempty']:
        raise ValueError('No usable kernel stacks were collected')
    if not record['stack_quality']['user_symbolized']:
        raise ValueError('No userspace frames could be resolved; profile is unqualified')
    coverage = unwind_coverage(record['samples'], events, record['stacks'],
                               record['user_symbols'], windows['training'])
    if coverage != record['training_unwind_coverage']:
        raise ValueError('Weighted unwind coverage differs from the raw training profile')
    if coverage['user_usable_fraction'] < .8 or coverage['possible_truncated_fraction'] > .2:
        raise ValueError('Training stack coverage is insufficient for qualified operator attribution')


def collect(args: argparse.Namespace) -> None:
    # BCC is a preinstalled, explicitly selected observer dependency. It is
    # loaded only by the privileged collector, never by the model subprocess.
    sys.path.insert(0, args.bcc_python_path)
    from bcc import BPF, PerfSWConfig, PerfType

    gate = load_json(args.gate)
    group = Path(gate['cgroup_path'])
    if (not str(group).startswith('/sys/fs/cgroup/system.slice/dg-native-iops-') or
            os.stat(group).st_ino != gate['cgroup_id']):
        raise ValueError('Profile target is not the owned system service')
    membership = fsutil.read_text(f'/proc/{gate["pid"]}/cgroup')
    if membership.strip() != '0::' + str(group).removeprefix('/sys/fs/cgroup'):
        raise ValueError('Target PID left its requested cgroup')
    program = profile_program(gate['cgroup_id'], args.mode)
    # Only the selected provider is attached. sched_switch/CPU callbacks do not
    # run together and the queue observer is launched in a separate trial.
    bpf = BPF(text=program)
    try:
        capture(args, bpf, program, gate, group, PerfType.SOFTWARE, PerfSWConfig.CPU_CLOCK)
    finally:
        bpf.cleanup()


def capture(args, bpf, program: str, gate: dict, group: Path, perf_type: int, cpu_clock: int) -> None:
    lost, overflow = [0], [False]
    data = bytearray()

    def on_interval(cpu, pointer, size):
        try:
            payload = interval_payload(ctypes.string_at(pointer, size))
        except ValueError:
            lost[0] += 1
            return
        if len(data) + len(payload) > args.event_capacity_bytes:
            overflow[0] = True
            return
        data.extend(payload)

    def on_lost(cpu, count):
        lost[0] += count

    if args.mode == 'cpu':
        bpf.attach_perf_event(ev_type=perf_type, ev_config=cpu_clock,
                              fn_name='cpu_sample', sample_freq=args.frequency)
    else:
        bpf['intervals'].open_perf_buffer(on_interval, lost_cb=on_lost, page_cnt=256)
    source_sha = hashlib.sha256(fsutil.read_binary(__file__)).hexdigest()
    meta = {'gate_pid': gate['pid'], 'cgroup_id': gate['cgroup_id'],
            'cgroup_path': str(group), 'collector_sha256': source_sha,
            'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
            'btf_sha256': hashlib.sha256(fsutil.read_binary('/sys/kernel/btf/vmlinux')).hexdigest(),
            'kernel_release': os.uname().release,
            'mode': args.mode, 'frequency_hz_per_cpu': args.frequency,
            'clock': 'CLOCK_MONOTONIC', 'event_capacity_bytes': args.event_capacity_bytes}
    ready = {'status': 'READY', **meta, 'ready_ns': time.monotonic_ns()}
    deadline = time.monotonic() + args.timeout
    maps = {}
    record = {}
    cgroup_removed = [None]

    def capture_maps(refresh: bool):
        try:
            members = fsutil.read_text(str(group / 'cgroup.procs')).split()
        except FileNotFoundError:
            # A --collect service can disappear after its last task exits but
            # before the parent publishes STOP. Markers and STOP still prove
            # whether the workload actually completed successfully.
            cgroup_removed[0] = time.monotonic_ns()
            return
        for text in members:
            if refresh or text not in maps:
                path = f'/proc/{text}/maps'
                if fsutil.exists(path):
                    try:
                        maps[text] = fsutil.read_text(path)
                    except FileNotFoundError:
                        pass  # A short child exited between procfs observations.

    try:
        bpf['enabled'][ctypes.c_int(0)] = ctypes.c_uint(1)
        save_json(args.ready, ready)
        next_maps, seen_markers = 0., -1
        while not fsutil.exists(args.stop):
            if time.monotonic() >= deadline:
                raise TimeoutError('Profile STOP was not received')
            if time.monotonic() >= next_maps:
                marker_count = bpf['marker_count'][ctypes.c_int(0)].value
                capture_maps(marker_count != seen_markers)
                seen_markers = marker_count
                next_maps = time.monotonic() + .25
            if args.mode == 'cpu':
                time.sleep(.05)
            else:
                bpf.perf_buffer_poll(timeout=50)
        stop = load_json(args.stop)
        if stop['exit_code']:
            raise RuntimeError('Profiled payload failed')
        bpf['enabled'][ctypes.c_int(0)] = ctypes.c_uint(0)
        # The owned system service has terminated before STOP. Drain delivery
        # only; no open target task can add a new interval at this point.
        if args.mode == 'offcpu':
            bpf.perf_buffer_poll(timeout=100)
        record = profile_snapshot(bpf, args.mode, bytes(data))
        samples, markers, stacks = record['samples'], record['markers'], record['stacks']
        users, kernels = set(), set()
        for sample in samples:
            users.add((sample['tgid'], sample['user']))
            kernels.add(sample['kernel'])
        for interval in INTERVAL.iter_unpack(data):
            users.add((interval[3], interval[7]))
            kernels.add(interval[8])
        record.update(process_maps=maps, **meta, gate=gate, ready=ready)
        # Symbolize after STOP. Preserve raw PCs and TGIDs so unavailable or
        # stale userspace mappings remain visible instead of guessed symbols.
        symbols = {str(pc): bpf.ksym(pc).decode(errors='replace')
                   for index in kernels for pc in stacks.get(str(index), [])}
        user_symbols = symbolize_users(stacks, users, maps)
        user_frames = {(pid, pc) for pid, index in users for pc in stacks.get(str(index), [])}
        resolved_users = sum(item['symbol'] != '??' for item in user_symbols.values())
        resolved_kernel = sum(not name.startswith(('0x', '[unknown]')) for name in symbols.values())
        row = load_json(stop['result_path'])
        windows = phase_windows(markers, row['iops_phase_markers'],
                                row['native_io']['fit']['after']['pid'])
        record = {**record, 'schema_version': 1, 'status': 'PASS', **meta, 'ready': ready,
                  'gate': gate, 'stop': stop, 'markers': markers,
                  'lost_events': lost[0], 'overflow': overflow[0],
                  'samples': samples,
                  'phase_cpu_samples': phase_samples(samples, args.frequency),
                  'sampling_coverage': sampling_coverage(phase_samples(samples, args.frequency), row)
                  if args.mode == 'cpu' else None,
                  'stacks': stacks, 'kernel_symbols': symbols, 'process_maps': maps,
                  'cgroup_removed_ns': cgroup_removed[0],
                  'user_symbols': user_symbols,
                  'stack_quality': {
                      'kernel_nonempty': any(stacks.get(str(index), []) for index in kernels),
                      'user_symbolized': resolved_users,
                      'user_unique_frames': len(user_frames),
                      'user_unknown_fraction': 1 - resolved_users / max(1, len(user_frames)),
                      'kernel_unique_frames': len(symbols),
                      'kernel_unknown_fraction': 1 - resolved_kernel / max(1, len(symbols)),
                      'user_stack_errors': sorted(index for pid, index in users if index < 0),
                      'kernel_stack_errors': sorted(index for index in kernels if index < 0)},
                  'mapping_semantics': 'Latest captured proc maps per PID at phase transitions; '
                  'unloaded/reused mappings may remain ambiguous; raw PCs are retained',
                  'phase_thread_time': interval_totals(data, windows) if args.mode == 'offcpu' else {},
                  'training_unwind_coverage': unwind_coverage(
                      samples, bytes(data), stacks, user_symbols, windows['training']),
                  'events': {'bytes': len(data), 'format': INTERVAL.format,
                             'sha256': hashlib.sha256(data).hexdigest()},
                  'offcpu_semantics': 'state&255==0 runnable/preempted; otherwise blocked sleep; '
                  'summed thread time, overlapping threads can exceed phase wall time'}
        validation_error = None
        try:
            validate_collection(record, row, gate, bytes(data))
        except ValueError as error:
            validation_error = error
            record.update(status='FAIL', validation_error=str(error))
        fsutil.write_binary(args.events, bytes(data))
        save_json(args.output, record)
        if validation_error is not None:
            raise validation_error
    except BaseException as error:
        bpf['enabled'][ctypes.c_int(0)] = ctypes.c_uint(0)
        record.update(profile_snapshot(bpf, args.mode, bytes(data)))
        record.update(status='FAIL', error=f'{type(error).__name__}: {error}', **meta,
                      gate=gate, ready=ready, process_maps=maps, lost_events=lost[0],
                      overflow=overflow[0], cgroup_removed_ns=cgroup_removed[0])
        fsutil.write_binary(args.events, bytes(data))
        save_json(args.output, record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect'])
    for name in ('candidate-root', 'gate', 'ready', 'stop', 'output', 'events', 'device'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mode', choices=['cpu', 'offcpu'], required=True)
    parser.add_argument('--frequency', type=int, default=99)
    parser.add_argument('--event-capacity-bytes', type=int, default=536870912)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--drain-seconds', type=float, default=10)
    parser.add_argument('--bcc-python-path', default='/usr/lib/python3/dist-packages')
    args = parser.parse_args()
    if not 1 <= args.frequency <= 999 or args.event_capacity_bytes < INTERVAL.size:
        parser.error('Invalid sampling frequency or bounded event capacity')
    collect(args)


if __name__ == '__main__':
    main()
