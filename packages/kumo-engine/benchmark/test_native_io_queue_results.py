# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib

import pytest
from native_io_queue_results import (
    ANCHOR,
    DIAGNOSTICS,
    END,
    EVENT,
    FINAL,
    FLUSH_TOKEN,
    ISSUE,
    LOGICAL_FLUSH,
    MARK,
    MARKERS,
    MERGE,
    REISSUE,
    REQUEUE,
    UNKNOWN,
    reconstruct,
    validate_collection,
)


def event(kind, ts, pointer=1, generation=None, cgroup=7, payload=0, operation=0,
          phase=0, entering=0, cpu=0):
    return [ts, pointer, ts if generation is None else generation, cgroup, payload,
            kind, operation, phase, entering, 100, cpu, 0]


def sample(extra=()):
    stamps = (10, 20, 30, 90, 100, 110, 120, 130, 140, 150, 160, 170)
    events = [event(ANCHOR, 5, pointer=0), event(END, 200, pointer=0)]
    clients = []
    for ts, (phase, entering) in zip(stamps, MARKERS, strict=True):
        events.append(event(MARK, ts, pointer=0, phase=phase, entering=entering))
        clients.append(dict(pid=100, phase_id=phase, entering=entering, before_ns=ts - 1, after_ns=ts + 1))
    events.extend(extra)
    sequences = {}
    events.sort(key=lambda row: row[0])
    for value in events:
        cpu = value[10]
        sequences[cpu] = sequences.get(cpu, 0) + 1
        value[11] = sequences[cpu]
    raw = b''.join(EVENT.pack(*value) for value in events)
    return raw, clients


def fixture_record():
    raw, clients = sample([event(ISSUE, 35), event(FINAL, 75, generation=35, payload=8192)])
    gate = dict(pid=90, cgroup_id=7, cgroup_path='/sys/fs/cgroup/system.slice/fixture.service')
    device = dict(path='/dev/nvme0n1', major=259, minor=0, io_stat_key='259:0')
    ready = dict(gate_pid=90, cgroup_id=7, cgroup_path=gate['cgroup_path'], bootstrap_ns=5,
                 device=device, collector_sha256='a' * 64, bpf_source_sha256='b' * 64,
                 btf_sha256='c' * 64, event_record_bytes=EVENT.size, event_capacity_bytes=10000)
    row = dict(native_io=dict(fit=dict(after=dict(pid=100))), iops_phase_markers=clients,
               matrix_spec=dict(result='/historical/result.json'),
               training_evidence=dict(pid_physical_read_bytes=8192))
    record = dict(schema_version=1, status='PASS', gate=gate, ready=ready, device=device,
                  collector_sha256='a' * 64, bpf_source_sha256='b' * 64, btf_sha256='c' * 64,
                  stop=dict(exit_code=0, result_path='/historical/result.json'),
                  diagnostics={name: 0 for name in DIAGNOSTICS}, event_capacity_bytes=10000,
                  final_kernel_state=dict(pending=0, request_entries=0, pending_flush=0,
                                          flush_entries=0, end_ns=200),
                  events=dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                              records=len(raw) // EVENT.size, kernel_records=len(raw) // EVENT.size,
                              format=EVENT.format), capture_end_ns=200,
                  derived=reconstruct(raw, 7, clients, 100, 200))
    return raw, row, gate, record


def test_analytic_overlap_and_phase_clip():
    raw, clients = sample([event(ISSUE, 25), event(FINAL, 75, generation=25, payload=8192),
                           event(ISSUE, 45, pointer=2, cgroup=8, cpu=1),
                           event(FINAL, 85, pointer=2, cgroup=8, generation=45, payload=4096, cpu=1)])
    result = reconstruct(raw, 7, clients, 100, 200)['phases']['training']
    assert result['device']['dispatched_depth_ns'] == {'1': 25, '2': 30, '0': 5}
    assert result['device']['dispatched_mean_depth'] == pytest.approx(85 / 60)
    assert result['device']['dispatched_idle_fraction'] == pytest.approx(5 / 60)
    assert result['target']['dispatched_mean_depth'] == .75
    assert result['target']['pending_lifecycle_at_start'] == 1
    assert result['target']['completed']['read']['carried_in_requests'] == 1
    assert result['target']['completed']['read']['latency_ns']['total'] == 50


def test_requeue_has_two_depth_definitions():
    raw, clients = sample([event(ISSUE, 35), event(REQUEUE, 45, generation=35),
                           event(REISSUE, 65, generation=35), event(FINAL, 75, generation=35)])
    phase = reconstruct(raw, 7, clients, 100, 200)['phases']['training']['target']
    assert phase['lifecycle_mean_depth'] == pytest.approx(40 / 60)
    assert phase['dispatched_mean_depth'] == pytest.approx(20 / 60)
    assert phase['completed']['read']['mean_latency_ns'] == 40


def test_foreign_flush_and_mixed_bounds():
    raw, clients = sample([event(ISSUE, 35, phase=9), event(FINAL, 75, generation=35, phase=9),
                           event(ISSUE, 45, pointer=2, cgroup=0, operation=2, phase=4),
                           event(FINAL, 55, pointer=2, cgroup=0, operation=2, phase=4, generation=45)])
    result = reconstruct(raw, 7, clients, 100, 200)
    phase = result['phases']['training']
    assert phase['target']['dispatched_mean_depth'] == 0
    assert phase['possible_target']['dispatched_mean_depth'] == pytest.approx(50 / 60)
    assert phase['device']['completed']['other']['payload_bytes']['minimum'] == 0
    assert result['counts']['mixed_requests'] == 1


def test_pointer_reuse_crosscpu_and_ties():
    raw, clients = sample([event(ISSUE, 35), event(FINAL, 55, generation=35, cpu=1),
                           event(ISSUE, 55, generation=55), event(FINAL, 75, generation=55, cpu=1)])
    chunks = [raw[index:index + EVENT.size] for index in range(0, len(raw), EVENT.size)]
    result = reconstruct(b''.join(reversed(chunks)), 7, clients, 100, 200)
    assert result['phases']['training']['target']['dispatched_mean_depth'] == pytest.approx(40 / 60)


def test_merge_and_carryout():
    raw, clients = sample([event(ISSUE, 35), event(MERGE, 55, generation=35),
                           event(ISSUE, 85, pointer=2), event(FINAL, 95, pointer=2, generation=85)])
    result = reconstruct(raw, 7, clients, 100, 200)
    assert result['counts']['merged'] == 1
    assert result['phases']['training']['target']['carried_out_requests'] == 1
    assert result['phases']['training']['target']['completed']['read']['latency_ns']['count'] == 0


def test_same_tick_phase_boundary_has_explicit_half_open_convention():
    raw, clients = sample([event(ISSUE, 35), event(FINAL, 90, generation=35)])
    result = reconstruct(raw, 7, clients, 100, 200)
    assert result['counts']['clock_tie_events'] == 1
    assert result['phases']['training']['target']['completed']['read']['latency_ns']['count'] == 0
    assert result['phases']['fit']['target']['completed']['read']['latency_ns']['count'] == 1


def flush_events():
    return [event(ISSUE, 35, operation=1),
            event(FLUSH_TOKEN, 74, generation=35, operation=1, payload=20, entering=2),
            event(FINAL, 75, generation=35, operation=1, payload=4096),
            event(LOGICAL_FLUSH, 95, generation=35, operation=1, payload=20)]


def test_flush_logical_completion_does_not_extend_device_work():
    raw, clients = sample(flush_events())
    result = reconstruct(raw, 7, clients, 100, 200)
    phase = result['phases']['training']['target']
    assert phase['dispatched_depth_ns'] == {'0': 20, '1': 40}
    assert phase['completed']['write']['payload_bytes']['total'] == 4096
    assert phase['completed']['write']['latency_ns']['total'] == 40
    assert result['counts']['issued'] == result['counts']['final'] == 1
    assert result['counts']['flush_tokens'] == result['counts']['logical_flush_callbacks'] == 1
    assert result['phases']['training']['logical_flush_callbacks'] == 0
    assert result['phases']['fit']['logical_flush_callbacks'] == 1


def test_flush_same_tick_crosscpu_respects_data_dependency():
    events = flush_events()
    events[1][0], events[1][10] = 75, 2
    events[2][10] = 2
    events[3][0], events[3][10] = 75, 1
    raw, clients = sample(events)
    result = reconstruct(raw, 7, clients, 100, 200)
    assert result['counts']['logical_flush_callbacks'] == 1
    assert result['phases']['training']['target']['completed']['write']['latency_ns']['count'] == 1


@pytest.mark.parametrize('mutation', [
    lambda e: e[1].__setitem__(4, 0),
    lambda e: e[1].__setitem__(4, 36),
    lambda e: e[1].__setitem__(8, 0),
    lambda e: e[3].__setitem__(4, 21),
    lambda e: e[3].__setitem__(2, 34),
    lambda e: e[3].__setitem__(3, 8),
    lambda e: e[3].__setitem__(6, 0),
    lambda e: e[3].__setitem__(8, 2),
    lambda e: e[2].__setitem__(4, 0),
    lambda e: e[3].__setitem__(0, 74),
    lambda e: e.pop(1),
    lambda e: e.pop(3),
    lambda e: e.insert(2, event(FLUSH_TOKEN, 74, generation=35, operation=1, payload=20, entering=2)),
    lambda e: e.append(event(ISSUE, 85, operation=1)),
    lambda e: e.append(event(LOGICAL_FLUSH, 96, generation=35, operation=1, payload=20)),
    lambda e: e.append(event(UNKNOWN, 96)),
])
def test_flush_token_corruption_fails(mutation):
    events = flush_events()
    mutation(events)
    raw, clients = sample(events)
    with pytest.raises(ValueError):
        reconstruct(raw, 7, clients, 100, 200)


@pytest.mark.parametrize('extra', [
    [event(ISSUE, 35), event(ISSUE, 40)],
    [event(FINAL, 45)],
    [event(UNKNOWN, 45)],
    [event(ISSUE, 35), event(FINAL, 75, generation=34)],
    [event(ISSUE, 35), event(REISSUE, 45, generation=35)],
    [event(ISSUE, 35), event(REQUEUE, 45, generation=35), event(FINAL, 75, generation=35)],
    [event(ISSUE, 35), event(MERGE, 55, generation=35, payload=1)],
    [event(ISSUE, 205)],
])
def test_invalid_lifetimes_fail(extra):
    raw, clients = sample(extra)
    with pytest.raises(ValueError):
        reconstruct(raw, 7, clients, 100, 200)


def test_sequence_loss_and_truncated_event():
    raw, clients = sample()
    with pytest.raises(ValueError, match='sequence'):
        reconstruct(raw[:EVENT.size] + raw[2 * EVENT.size:], 7, clients, 100, 200)
    with pytest.raises(ValueError, match='Truncated'):
        reconstruct(raw[:-1], 7, clients, 100, 200)


def test_saved_record_replays_without_historical_reads():
    raw, row, gate, record = fixture_record()
    assert validate_collection(record, row, gate, raw) == record['derived']


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(status='FAIL'),
    lambda r: r['ready'].update(gate_pid=91),
    lambda r: r['stop'].update(exit_code=True),
    lambda r: r['stop'].update(result_path='/other/result.json'),
    lambda r: r['final_kernel_state'].update(pending_flush=1),
    lambda r: r['final_kernel_state'].update(flush_entries=1),
    lambda r: r['final_kernel_state'].update(pending=False),
    lambda r: r['final_kernel_state'].update(end_ns=201),
    lambda r: r['diagnostics'].update(perf_lost=1),
    lambda r: r.update(diagnostics={}),
    lambda r: r['events'].update(kernel_records=999),
    lambda r: r['events'].update(sha256='0' * 64),
    lambda r: r['events'].update(format='wrong'),
    lambda r: r.update(collector_sha256='0' * 64),
    lambda r: r['device'].update(major=8),
    lambda r: r.update(capture_end_ns=201),
    lambda r: r['derived']['phases']['training']['target'].update(dispatched_mean_depth=2),
])
def test_saved_corruption_fails(mutation):
    raw, row, gate, record = fixture_record()
    record = copy.deepcopy(record)
    mutation(record)
    with pytest.raises(ValueError):
        validate_collection(record, row, gate, raw)
