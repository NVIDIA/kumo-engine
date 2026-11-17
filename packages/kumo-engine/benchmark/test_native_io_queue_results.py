# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib

import pytest
from native_io_queue_results import (
    ANCHOR,
    DIAGNOSTICS,
    EMPTY_FLUSH_DONE,
    EMPTY_FLUSH_ENTRY,
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
    unknown_details,
    validate_collection,
    validate_empty_protocol,
    validate_unknown,
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


def test_unknown_metadata_remains_fatal():
    value = event(UNKNOWN, 35, pointer=0xffff0000, generation=12, cgroup=0x1201,
                  operation=1, phase=2, entering=0xfffffff4)
    raw, clients = sample([value])
    details = unknown_details(raw, 1)
    assert details == [dict(timestamp_ns=35, pointer=0xffff0000, start_time_ns=12,
                            cmd_flags=0x1201, rq_flags=2, remaining_bytes=0,
                            operation=1, kernel_stack_id=-12, pid=100, cpu=0, sequence=5)]
    with pytest.raises(ValueError, match='Left-censored/untracked'):
        reconstruct(raw, 7, clients, 100, 200)


@pytest.mark.parametrize('version', [None, 0, True, 2])
def test_unknown_requires_explicit_encoding(version):
    with pytest.raises(ValueError, match='encoding'):
        unknown_details(b'', version)


@pytest.mark.parametrize('field,value', [(3, 1 << 32), (4, 1 << 32), (6, 2), (8, 128)])
def test_unknown_metadata_width_and_stack_bounds(field, value):
    item = event(UNKNOWN, 30, cgroup=1, operation=1, entering=0)
    item[field] = value
    with pytest.raises(ValueError, match='UNKNOWN'):
        unknown_details(EVENT.pack(*item), 1)


def test_unknown_saved_raw_binding():
    raw, row, gate, record = fixture_record()
    record['unknown_encoding_version'] = record['ready']['unknown_encoding_version'] = 1
    record['unknown_provenance'] = dict(encoding_version=1, stack_capacity=128,
                                        events=[], stacks={}, symbols={}, stack_errors=[])
    validate_collection(record, row, gate, raw)
    record['unknown_provenance']['events'] = [dict(operation=1)]
    with pytest.raises(ValueError, match='UNKNOWN provenance'):
        validate_collection(record, row, gate, raw)


def test_unknown_saved_stack_binding():
    raw, _ = sample([event(UNKNOWN, 3, cgroup=1, operation=1, entering=7)])
    record = dict(ready=dict(unknown_encoding_version=1), unknown_encoding_version=1,
                  unknown_provenance=dict(encoding_version=1, stack_capacity=128,
                                          events=unknown_details(raw, 1), stacks={'7': [1234]},
                                          symbols={'1234': 'blk_mq_end_request'}, stack_errors=[]))
    validate_unknown(record, raw)
    record['unknown_provenance']['stacks']['7'] = []
    with pytest.raises(ValueError, match='no kernel PCs'):
        validate_unknown(record, raw)
    record['unknown_provenance']['stacks']['7'] = [1234]
    record['unknown_provenance']['stacks']['8'] = [5678]
    with pytest.raises(ValueError, match='referenced stacks'):
        validate_unknown(record, raw)


def empty_pair(start=35, end=75, allocation=12, pointer=8):
    return [event(EMPTY_FLUSH_ENTRY, start, pointer=pointer, generation=start, cgroup=0,
                  operation=1, payload=allocation, phase=0x100, entering=0x40001, cpu=1),
            event(EMPTY_FLUSH_DONE, end, pointer=pointer, generation=start, cgroup=0,
                  operation=1, payload=allocation, phase=0x100, entering=0x801)]


@pytest.mark.parametrize('start,end', [(35, 75), (35, 35), (3, 4)])
def test_empty_flush_sync_async_never_changes_physical_metrics(start, end):
    physical = [event(ISSUE, 40), event(FINAL, 70, generation=40, payload=4096)]
    raw, clients = sample(physical)
    baseline = reconstruct(raw, 7, clients, 100, 200)
    raw, clients = sample(physical + empty_pair(start, end, allocation=min(12, start)))
    result = reconstruct(raw, 7, clients, 100, 200)
    assert result['counts']['empty_flush_entries'] == result['counts']['empty_logical_callbacks'] == 1
    for name, phase in baseline['phases'].items():
        for scope in ('device', 'target', 'possible_target', 'mixed_or_unchecked', 'foreign_or_unknown'):
            assert result['phases'][name][scope] == phase[scope]


def test_empty_flush_allocation_reuse_after_completion():
    raw, clients = sample(empty_pair() + empty_pair(80, 85, allocation=79))
    assert reconstruct(raw, 7, clients, 100, 200)['counts']['empty_logical_callbacks'] == 2


@pytest.mark.parametrize('mutation', [
    lambda e: e.pop(0),
    lambda e: e.pop(),
    lambda e: e.append(e[1].copy()),
    lambda e: e.insert(1, e[0].copy()),
    lambda e: e[0].__setitem__(2, 34),
    lambda e: e[0].__setitem__(4, 0),
    lambda e: e[0].__setitem__(4, 36),
    lambda e: e[0].__setitem__(8, 0x60001),
    lambda e: e[0].__setitem__(8, 0x20001),
    lambda e: e[0].__setitem__(7, 0x102),
    lambda e: e[0].__setitem__(6, 0),
    lambda e: e[1].__setitem__(4, 13),
    lambda e: e[1].__setitem__(2, 36),
    lambda e: e[1].__setitem__(7, 0),
    lambda e: e[1].__setitem__(8, 1),
    lambda e: e[1].__setitem__(8, 0x40801),
    lambda e: e[1].__setitem__(3, 7),
    lambda e: e.insert(1, event(ISSUE, 45, pointer=8)),
])
def test_empty_flush_invalid_lifecycles_fail(mutation):
    values = empty_pair()
    mutation(values)
    raw, clients = sample(values)
    with pytest.raises(ValueError):
        reconstruct(raw, 7, clients, 100, 200)


def test_separate_physical_flush_still_occupies_device():
    physical = [event(ISSUE, 45, pointer=9, operation=2),
                event(FINAL, 65, pointer=9, operation=2, generation=45)]
    raw, clients = sample(empty_pair() + physical)
    training = reconstruct(raw, 7, clients, 100, 200)['phases']['training']
    assert training['device']['completed']['other']['latency_ns']['count'] == 1
    assert training['device']['dispatched_depth_ns'] == {'0': 40, '1': 20}


def empty_protocol():
    return dict(version=1, function='blk_insert_flush', signature='bool(struct request *)',
                function_type_id=9, prototype_type_id=8, request_type_id=1, bool_type_id=5,
                enums=dict(REQ_OP_WRITE=1, __REQ_PREFLUSH=18, __REQ_FUA=17, __REQ_SYNC=11, __RQF_FLUSH_SEQ=1))


def test_empty_flush_saved_protocol_binding():
    raw, _ = sample(empty_pair())
    record = dict(empty_flush_protocol=empty_protocol(), ready=dict(empty_flush_protocol=empty_protocol()))
    validate_empty_protocol(record, raw)
    record['empty_flush_protocol']['enums']['__REQ_PREFLUSH'] = 19
    with pytest.raises(ValueError, match='signature/enums'):
        validate_empty_protocol(record, raw)
    with pytest.raises(ValueError, match='BTF protocol binding'):
        validate_empty_protocol(dict(ready={}), raw)


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
