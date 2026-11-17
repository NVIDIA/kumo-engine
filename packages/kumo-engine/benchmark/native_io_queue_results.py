# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline reconstruction of attributed, timestamped block-request lifetimes."""

import argparse
import hashlib
import heapq
import json
import math
import struct
from array import array

from io_utils import fsutil

EVENT = struct.Struct('<QQQQQIIIIIIQ')
ISSUE, FINAL, REQUEUE, REISSUE, MERGE, MARK, ANCHOR, UNKNOWN = range(1, 9)
END, LOGICAL_FLUSH, FLUSH_TOKEN = 9, 10, 11
FLUSH_SEQ_MASK = 2  # Collector rejects BTF layouts where __RQF_FLUSH_SEQ != 1.
UNKNOWN_ENCODING = 1
UNKNOWN_STACK_CAPACITY = 128
PHASES = {1: 'fit', 2: 'training', 3: 'predict', 4: 'repeat', 5: 'reload', 6: 'e2e'}
MARKERS = [(6, 1), (1, 1), (2, 1), (2, 0), (1, 0), (3, 1), (3, 0), (6, 0),
           (4, 1), (4, 0), (5, 1), (5, 0)]
SCOPES = ('device', 'target', 'mixed_or_unchecked', 'foreign_or_unknown', 'possible_target')
DIAGNOSTICS = ('probe_read_failure', 'missing_read_write_bio', 'request_map_full',
               'invalid_reissue', 'completion_error_or_partial_merge',
               'request_delete_failure', 'perf_submit_failure', 'sequence_map_missing',
               'invalid_flush_token', 'flush_token_delete_failure',
               'perf_lost', 'ram_overflow', 'record_size')


def unknown_details(raw: bytes, version: int) -> list[dict]:
    """Decode diagnostic-only fields; old zero-metadata UNKNOWNs lack a version."""
    if type(version) is not int or version != UNKNOWN_ENCODING:
        raise ValueError('Unsupported UNKNOWN diagnostic encoding')
    if len(raw) % EVENT.size:
        raise ValueError('Truncated event record')
    details = []
    for event in EVENT.iter_unpack(raw):
        if event[5] != UNKNOWN:
            continue
        ts, pointer, start, cmd, remaining, _, operation, flags, stack, pid, cpu, sequence = event
        if cmd > 0xffffffff or remaining > 0xffffffff or operation != cmd & 255:
            raise ValueError('Invalid UNKNOWN request metadata widths or operation')
        signed_stack = stack if stack < 0x80000000 else stack - 0x100000000
        if signed_stack >= UNKNOWN_STACK_CAPACITY:
            raise ValueError('UNKNOWN stack ID exceeds bounded map')
        details.append(dict(timestamp_ns=ts, pointer=pointer, start_time_ns=start,
                            cmd_flags=cmd, rq_flags=flags, remaining_bytes=remaining,
                            operation=operation, kernel_stack_id=signed_stack,
                            pid=pid, cpu=cpu, sequence=sequence))
    return details


def validate_unknown(record: dict, raw: bytes) -> None:
    """Check versioned provenance without interpreting it as a valid lifecycle."""
    ready = record['ready']
    if 'unknown_encoding_version' not in record and 'unknown_encoding_version' not in ready:
        if 'unknown_provenance' in record:
            raise ValueError('Unversioned UNKNOWN provenance')
        return  # Historical captures do not claim diagnostic metadata.
    version = record.get('unknown_encoding_version')
    details = unknown_details(raw, version)
    value = record['unknown_provenance']
    if (type(ready.get('unknown_encoding_version')) is not int
            or ready['unknown_encoding_version'] != version
            or type(value['encoding_version']) is not int or value['encoding_version'] != version
            or type(value['stack_capacity']) is not int or value['stack_capacity'] != UNKNOWN_STACK_CAPACITY
            or value['events'] != details or 'unknown_provenance_error' in record):
        raise ValueError('UNKNOWN provenance version or raw metadata differs')
    ids = {event['kernel_stack_id'] for event in details}
    stacks = value['stacks']
    if (set(stacks) != {str(index) for index in ids if index >= 0}
            or value['stack_errors'] != sorted(index for index in ids if index < 0)):
        raise ValueError('UNKNOWN referenced stacks or errors differ')
    if any(not isinstance(frames, list) or not frames for frames in stacks.values()):
        raise ValueError('Successful UNKNOWN stack ID has no kernel PCs')
    if any(type(pc) is not int or pc <= 0 for frames in stacks.values() for pc in frames):
        raise ValueError('Invalid UNKNOWN kernel PC')
    if (set(value['symbols']) != {str(pc) for frames in stacks.values() for pc in frames}
            or any(not isinstance(symbol, str) for symbol in value['symbols'].values())):
        raise ValueError('UNKNOWN kernel symbol coverage differs')


def membership(cgroup: int, target: int, flags: int) -> tuple[int, ...]:
    pure = int(cgroup == target and flags & 7 == 0)
    mixed = int(bool(flags & 3))
    possible = int(bool(pure or flags & 6 or (flags & 1 and flags & 8)))
    return (1, pure, mixed, int(not pure and not mixed), possible)


def ordered_events(raw: bytes):
    """Merge CPU timelines without materializing millions of Python tuples."""
    if len(raw) % EVENT.size:
        raise ValueError('Truncated event record')
    cpus = {}
    for offset in range(0, len(raw), EVENT.size):
        event = EVENT.unpack_from(raw, offset)
        cpus.setdefault(event[10], array('Q')).append(offset)
    timelines = []
    for offsets in cpus.values():
        by_sequence = sorted(offsets, key=lambda offset: EVENT.unpack_from(raw, offset)[11])
        if any(EVENT.unpack_from(raw, offset)[11] != index
               for index, offset in enumerate(by_sequence, 1)):
            raise ValueError('Lost or duplicate per-CPU event sequence')
        offsets = array('Q', sorted(offsets, key=lambda offset: event_order(EVENT.unpack_from(raw, offset))))
        timelines.append((EVENT.unpack_from(raw, offset) for offset in offsets))
    yield from heapq.merge(*timelines, key=event_order)


def event_order(event: tuple) -> tuple:
    # Generation orders an old final before pointer reuse at the same clock
    # tick. Other simultaneous transitions have zero intervening residence.
    # A logical flush callback depends on the earlier DATA completion even
    # when another CPU observes both within one finite-resolution clock tick.
    return (event[0], event[1], event[2], int(event[5] == LOGICAL_FLUSH), event[10], event[11])


def histogram() -> dict:
    return dict(count=0, total=0, minimum=None, maximum=None, log2={})


def observe(hist: dict, value: int) -> None:
    if value < 0:
        raise ValueError('Negative duration or payload')
    hist['count'] += 1
    hist['total'] += value
    hist['minimum'] = value if hist['minimum'] is None else min(value, hist['minimum'])
    hist['maximum'] = value if hist['maximum'] is None else max(value, hist['maximum'])
    key = str(value.bit_length())  # bucket0 is zero; bucketN covers [2**(N-1),2**N).
    hist['log2'][key] = hist['log2'].get(key, 0) + 1


def scope_state() -> dict:
    return dict(lifecycle_depth_ns={}, dispatched_depth_ns={},
                completed={name: dict(latency_ns=histogram(), payload_bytes=histogram(), carried_in_requests=0)
                           for name in ('read', 'write', 'other')})


def phase_state(start: int, depths: list[int]) -> dict:
    phase = dict(start_ns=start, logical_flush_callbacks=0, **{name: scope_state() for name in SCOPES})
    for index, name in enumerate(SCOPES):
        phase[name].update(pending_lifecycle_at_start=depths[2 * index],
                           pending_dispatched_at_start=depths[2 * index + 1])
    return phase


def close_phase(phase: dict, end: int, depths: list[int], requests: dict, target: int) -> None:
    phase['end_ns'] = end
    for index, name in enumerate(SCOPES):
        carried = sum(membership(request[1], target, request[4])[index]
                      for request in requests.values() if request[0] >= phase['start_ns'])
        phase[name].update(pending_lifecycle_at_end=depths[2 * index],
                           pending_dispatched_at_end=depths[2 * index + 1], carried_out_requests=carried)


def integrate(phase: dict, depths: list[int], elapsed: int) -> None:
    if elapsed < 0:
        raise ValueError('Nonmonotonic event timeline')
    if elapsed == 0:
        return
    for scope_index, scope in enumerate(SCOPES):
        offset = scope_index * 2
        for position, name in enumerate(('lifecycle_depth_ns', 'dispatched_depth_ns')):
            hist = phase[scope][name]
            key = str(depths[offset + position])
            hist[key] = hist.get(key, 0) + elapsed


def finalize(phases: dict) -> None:
    for phase in phases.values():
        duration = phase['end_ns'] - phase['start_ns']
        if duration <= 0:
            raise ValueError('Nonpositive phase clock')
        phase['seconds'] = duration / 1e9
        for scope in SCOPES:
            data = phase[scope]
            for name in ('lifecycle', 'dispatched'):
                hist = data[name + '_depth_ns']
                if sum(hist.values()) != duration:
                    raise ValueError('Depth timeline does not cover the phase')
                data[name + '_mean_depth'] = sum(int(k) * v for k, v in hist.items()) / duration
                data[name + '_idle_fraction'] = hist.get('0', 0) / duration
                data[name + '_idle_ns'] = hist.get('0', 0)
                data[name + '_active_ns'] = duration - hist.get('0', 0)
            for stats in data['completed'].values():
                latency = stats['latency_ns']
                stats['mean_latency_ns'] = latency['total'] / latency['count'] if latency['count'] else None
                stats['requests_per_second'] = latency['count'] * 1e9 / duration


def reconstruct(raw: bytes, target_cgroup: int, client_marks: list[dict],
                model_pid: int, capture_end_ns: int) -> dict:
    if len(client_marks) != 12:
        raise ValueError('Expected twelve client markers')
    requests, phases, active, tokens = {}, {}, {}, {}
    depths, previous, anchor, marker_index, ended = [0] * (2 * len(SCOPES)), None, None, 0, False
    counts = dict(issued=0, final=0, merged=0, requeues=0, reissues=0, bootstrap_unknown=0,
                  mixed_requests=0, unchecked_bio_tails=0, unattributed_operations=0, clock_tie_events=0,
                  flush_tokens=0, logical_flush_callbacks=0)
    for event in ordered_events(raw):
        ts, pointer, generation, cgroup, payload, kind, operation, phase_id, entering, pid, _, _ = event
        if ended or ts <= 0 or ts > capture_end_ns or kind not in range(1, 12):
            raise ValueError('Invalid event time or kind')
        counts['clock_tie_events'] += int(previous == ts)
        if previous is not None and anchor is not None:
            for phase in active.values():
                integrate(phase, depths, ts - previous)
        previous = ts
        if kind == UNKNOWN:
            if anchor is not None:
                raise ValueError('Left-censored/untracked request after bootstrap')
            counts['bootstrap_unknown'] += 1
            continue
        if kind == ANCHOR:
            if anchor is not None or requests or tokens or any(depths):
                raise ValueError('Bootstrap must establish one empty-device epoch')
            anchor = ts
            phases['observation'] = active['observation'] = phase_state(ts, depths)
            continue
        if kind == END:
            if (anchor is None or requests or tokens or any(depths) or set(active) != {'observation'}
                    or ts != capture_end_ns or marker_index != 12):
                raise ValueError('Invalid final drained capture boundary')
            close_phase(active.pop('observation'), ts, depths, requests, target_cgroup)
            ended = True
            continue
        if kind == MARK:
            if anchor is None or marker_index >= len(MARKERS):
                raise ValueError('Marker outside anchored measurement')
            if (phase_id, entering) != MARKERS[marker_index] or pid != model_pid:
                raise ValueError('Unexpected marker protocol or model PID')
            client = client_marks[marker_index]
            if (client['phase_id'], client['entering'], client['pid']) != (phase_id, entering, pid):
                raise ValueError('Client and kernel marker identities differ')
            if not client['before_ns'] <= ts <= client['after_ns']:
                raise ValueError('Kernel marker outside client clock bracket')
            name = PHASES[phase_id]
            if entering:
                phases[name] = active[name] = phase_state(ts, depths)
            else:
                close_phase(active.pop(name), ts, depths, requests, target_cgroup)
            marker_index += 1
            continue
        if phase_id & ~15:
            raise ValueError('Unsupported BIO attribution flags')
        if kind == LOGICAL_FLUSH:
            token = tokens.get(pointer)
            if (pointer in requests or token is None or not token[5]
                    or token[:5] != [generation, cgroup, operation, phase_id, payload]
                    or entering & FLUSH_SEQ_MASK):
                raise ValueError('Logical flush callback lacks its exact completed DATA token')
            del tokens[pointer]
            counts['logical_flush_callbacks'] += 1
            for phase in active.values():
                phase['logical_flush_callbacks'] += 1
            continue
        if entering and kind != FLUSH_TOKEN:
            raise ValueError('Unexpected request event metadata')
        members = membership(cgroup, target_cgroup, phase_id)
        if kind == ISSUE:
            if pointer in requests or pointer in tokens or generation != ts:
                raise ValueError('Duplicate request or invalid generation')
            requests[pointer] = [generation, cgroup, operation, True, phase_id]
            for index, member in enumerate(members):
                depths[index * 2] += member
                depths[index * 2 + 1] += member
            counts['issued'] += 1
            counts['mixed_requests'] += bool(phase_id & 1)
            counts['unchecked_bio_tails'] += bool(phase_id & 2)
            counts['unattributed_operations'] += bool(phase_id & 4)
            continue
        request = requests.get(pointer)
        if request is None or request[:3] != [generation, cgroup, operation] or request[4] != phase_id:
            raise ValueError('Request identity/lifecycle mismatch')
        if kind == FLUSH_TOKEN:
            if (pointer in tokens or operation != 1 or not request[3]
                    or not entering & FLUSH_SEQ_MASK or not 0 < payload <= generation):
                raise ValueError('Invalid flush DATA token or start_time_ns')
            tokens[pointer] = [generation, cgroup, operation, phase_id, payload, False]
            counts['flush_tokens'] += 1
            continue
        if pointer in tokens and kind != FINAL:
            raise ValueError('Flush DATA token must precede its physical final')
        if kind == REQUEUE:
            if not request[3]:
                raise ValueError('Duplicate requeue')
            request[3] = False
            for index, member in enumerate(members):
                depths[index * 2 + 1] -= member
            counts['requeues'] += 1
        elif kind == REISSUE:
            if request[3]:
                raise ValueError('Reissue without requeue')
            request[3] = True
            for index, member in enumerate(members):
                depths[index * 2 + 1] += member
            counts['reissues'] += 1
        elif kind in (FINAL, MERGE):
            if kind == FINAL and not request[3]:
                raise ValueError('Final completion while requeued')
            if kind == MERGE and payload:
                raise ValueError('Partial completion followed by merge is unsupported')
            if pointer in tokens:
                if operation != 1 or not payload:
                    raise ValueError('Flush DATA physical completion must have WRITE payload')
                tokens[pointer][5] = True
            for index, member in enumerate(members):
                depths[index * 2] -= member
                depths[index * 2 + 1] -= member * int(request[3])
            del requests[pointer]
            counts['final' if kind == FINAL else 'merged'] += 1
            if kind == FINAL and anchor is not None:
                op = 'read' if operation == 0 else 'write' if operation == 1 else 'other'
                for phase in active.values():
                    for scope, member in zip(SCOPES, members, strict=True):
                        if member:
                            stats = phase[scope]['completed'][op]
                            observe(stats['latency_ns'], ts - generation)
                            observe(stats['payload_bytes'], payload)
                            stats['carried_in_requests'] += int(generation < phase['start_ns'])
        else:
            raise ValueError('Unexpected request event')
        if min(depths) < 0:
            raise ValueError('Negative queue depth')
    if anchor is None or marker_index != 12 or len(client_marks) != 12:
        raise ValueError('Missing bootstrap or twelve phase markers')
    if requests or tokens or any(depths) or active or not ended:
        raise ValueError('Undrained request/phase state')
    if counts['issued'] != counts['final'] + counts['merged']:
        raise ValueError('Request reconciliation failed')
    if counts['flush_tokens'] != counts['logical_flush_callbacks']:
        raise ValueError('Flush token reconciliation failed')
    finalize(phases)
    return dict(phases=phases, counts=counts, bootstrap_ns=anchor,
                latency_definition='First postmerge issue to final callback, including requeue delay; completion-phase assignment',
                flush_definition='Verified flush DATA is counted once at physical completion; its later token-matched logical callback adds no device depth, payload or latency',
                idle_definition='No dispatched tracked physical-device requests, including other operations and foreign cgroups; not NAND internal idle',
                target_definition='Pure target is lower occupancy bound; possible_target adds mixed BIOs containing target and unknown BIO tails/origins as conservative upper bound. No journal-write causality inferred',
                clock_definition='Observed kernel timestamps, not perfect physical ordering; ties use request pointer/generation, physical-before-logical flush, then CPU sequence. Marker pointer0 precedes same-tick requests (half-open phase convention); zero-time transitions add no occupancy area',
                histogram_definition='Depth keys are exact depths with nanoseconds; log2 bucket0=zero, bucketN=[2**(N-1),2**N)')


def validate_collection(record: dict, row: dict, gate: dict, events_bytes: bytes) -> dict:
    if record['schema_version'] != 1 or record['status'] != 'PASS':
        raise ValueError('Queue collector did not pass')
    for name in ('cgroup_id', 'cgroup_path', 'pid'):
        if record['gate'][name] != gate[name]:
            raise ValueError('Target gate identity differs')
    ready = record['ready']
    if (ready['gate_pid'] != gate['pid'] or ready['cgroup_id'] != gate['cgroup_id']
            or ready['cgroup_path'] != gate['cgroup_path']):
        raise ValueError('Collector READY identity differs')
    if (set(record['diagnostics']) != set(DIAGNOSTICS)
            or any(type(value) is not int or value != 0 for value in record['diagnostics'].values())):
        raise ValueError('Collector loss/probe/overflow diagnostic')
    stop = record['stop']
    if (type(stop['exit_code']) is not int or stop['exit_code'] != 0
            or stop['result_path'] != row['matrix_spec']['result']):
        raise ValueError('STOP must prove successful exit for the exact workload result')
    state = record.get('final_kernel_state', {})
    if (set(state) != {'pending', 'request_entries', 'pending_flush', 'flush_entries', 'end_ns'}
            or any(type(state.get(key)) is not int or state[key] != 0
                   for key in ('pending', 'request_entries', 'pending_flush', 'flush_entries'))
            or type(state.get('end_ns')) is not int or state['end_ns'] != record['capture_end_ns']):
        raise ValueError('Final kernel request/token state was not exactly drained')
    oracle = record['events']
    if (oracle['sha256'] != hashlib.sha256(events_bytes).hexdigest()
            or oracle['bytes'] != len(events_bytes) or oracle['records'] * EVENT.size != len(events_bytes)
            or oracle['kernel_records'] != oracle['records'] or oracle['format'] != EVENT.format
            or len(events_bytes) > record['event_capacity_bytes'] or ready['event_record_bytes'] != EVENT.size
            or ready['event_capacity_bytes'] != record['event_capacity_bytes']):
        raise ValueError('Raw event hash/count mismatch')
    validate_unknown(record, events_bytes)
    for name in ('collector_sha256', 'bpf_source_sha256', 'btf_sha256'):
        if (record[name] != ready[name] or len(record[name]) != 64
                or any(char not in '0123456789abcdef' for char in record[name])):
            raise ValueError('Source/BTF identity differs')
    device = record['device']
    if device != ready['device'] or device['io_stat_key'] != f'{device["major"]}:{device["minor"]}':
        raise ValueError('Physical-device identity differs')
    model_pid = row['native_io']['fit']['after']['pid']
    result = reconstruct(events_bytes, gate['cgroup_id'], row['iops_phase_markers'],
                         model_pid, record['capture_end_ns'])
    if result != record['derived'] or result['bootstrap_ns'] != ready['bootstrap_ns']:
        raise ValueError('Saved queue results differ from exact event replay')
    training_reads = result['phases']['training']['possible_target']['completed']['read']['latency_ns']['count']
    if row['training_evidence']['pid_physical_read_bytes'] > 0 and training_reads == 0:
        raise ValueError('Positive process training reads but no possible target completions')
    for name in ('fit', 'training', 'predict', 'repeat', 'reload', 'e2e'):
        seconds = result['phases'][name]['seconds']
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid physical phase clock')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True)
    parser.add_argument('--events', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    record = json.loads(fsutil.read_text(args.record))
    row = json.loads(fsutil.read_text(args.result))
    result = validate_collection(record, row, record['gate'], fsutil.read_binary(args.events))
    print(json.dumps(dict(status='QUEUE_REPLAY_PASS', phases=list(result['phases']))))


if __name__ == '__main__':
    main()
