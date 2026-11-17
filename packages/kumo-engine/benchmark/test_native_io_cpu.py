# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
from native_io_cpu import (
    INTERVAL,
    MARKERS,
    interval_payload,
    interval_totals,
    mapped_address,
    phase_samples,
    phase_windows,
    profile_program,
    sampling_coverage,
    unwind_coverage,
    validate_collection,
)


def markers():
    kernel, client, mask = [], [], 0
    for number, (phase, entering) in enumerate(MARKERS, 1):
        stamp = number * 100
        mask = mask | (1 << phase) if entering else mask & ~(1 << phase)
        kernel.append({'ns': stamp, 'pid': 42, 'tid': 42, 'phase': phase,
                       'entering': entering, 'mask': mask})
        client.append({'before_ns': stamp - 1, 'after_ns': stamp + 1,
                       'pid': 42, 'phase_id': phase, 'entering': entering})
    return kernel, client


def test_marker_windows_are_nested_and_clock_bracketed():
    kernel, client = markers()
    windows = phase_windows(kernel, client, 42)
    assert windows['training'] == (300, 400)
    assert windows['fit'] == (200, 500)
    assert windows['e2e'] == (100, 800)
    kernel[2]['ns'] = 310
    with pytest.raises(ValueError, match='clock'):
        phase_windows(kernel, client, 42)


@pytest.mark.parametrize('field,value', [('pid', 43), ('tid', 43), ('phase', 5), ('mask', 0)])
def test_reject_marker_identity_and_phase_corruption(field, value):
    kernel, client = markers()
    kernel[2][field] = value
    with pytest.raises(ValueError):
        phase_windows(kernel, client, 42)


def wait(start, end, state):
    return INTERVAL.pack(start, end, state, 42, 43, 0, 0, -1, 5, b'worker')


def test_offcpu_classifies_preemption_and_clips_phase_edges():
    events = wait(10, 30, 1) + wait(20, 50, 0) + wait(40, 60, 256)
    totals = interval_totals(events, {'training': (15, 55)})
    assert totals == {'training': {'blocked_sleep_ns': 15, 'runnable_or_preempted_ns': 45}}
    # Summed thread time exceeds wall time when several threads overlap.
    assert sum(totals['training'].values()) > 55 - 15


def test_truncated_interval_is_not_silently_accepted():
    with pytest.raises(ValueError, match='Truncated'):
        interval_totals(wait(10, 20, 1)[:-1], {'training': (0, 30)})


def test_perf_alignment_is_checked_and_not_written_as_payload():
    raw = wait(10, 20, 1)
    assert interval_payload(raw) == raw
    assert interval_payload(raw + b'\0' * 4) == raw
    assert interval_payload(raw + b'\x01\0\x58\0') == raw
    changed = raw[:-1] + b'x'
    assert interval_payload(changed + b'\xff' * 4) == changed


@pytest.mark.parametrize('extra_bytes', [1, 2, 3, 5, 6, 7, 8])
def test_perf_alignment_rejects_unexpected_widths(extra_bytes):
    with pytest.raises(ValueError, match='length'):
        interval_payload(wait(10, 20, 1) + b'\0' * extra_bytes)


def test_captured_maps_preserve_elf_file_offsets():
    maps = '1000-2000 r-xp 00003000 00:00 1 /library.so\n'
    assert mapped_address(0x1200, maps) == ('/library.so', 0x3200)
    assert mapped_address(0x2000, maps) is None


def capture():
    kernel, client = markers()
    gate = {'pid': 10, 'cgroup_id': 20, 'cgroup_path': '/scope'}
    meta = {'gate_pid': 10, 'cgroup_id': 20, 'cgroup_path': '/scope',
            'collector_sha256': 'a' * 64,
            'program_sha256': hashlib.sha256(profile_program(20, 'cpu').encode()).hexdigest(),
            'mode': 'cpu',
            'btf_sha256': 'b' * 64, 'kernel_release': '6.17', 'event_capacity_bytes': 4096}
    row = {'iops_phase_markers': client, 'native_io': {'fit': {'after': {'pid': 42}}},
           'matrix_spec': {'result': '/result'}, 'work': {'training_interval': {
               'resources_before': {'cpu': {'usage_usec': 0}},
               'resources_after': {'cpu': {'usage_usec': 101010}}}}}
    samples = [{'count': 10, 'mask': 70, 'tgid': 42, 'user': 1, 'kernel': 2}]
    stacks, symbols = {'1': [100], '2': [200]}, {'42:100': {'symbol': 'native_function'}}
    record = {'status': 'PASS', **meta, 'ready': {'status': 'READY', **meta}, 'gate': gate,
              'stop': {'exit_code': 0, 'result_path': '/result'},
              'markers': kernel, 'events': {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest(),
                                            'format': INTERVAL.format},
              'failures': [0, 0, 0], 'lost_events': 0, 'overflow': False,
              'samples': samples, 'frequency_hz_per_cpu': 99,
              'phase_cpu_samples': phase_samples(samples, 99),
              'sampling_coverage': sampling_coverage(phase_samples(samples, 99), row),
              'stacks': stacks, 'user_symbols': symbols,
              'training_unwind_coverage': unwind_coverage(samples, b'', stacks, symbols, (300, 400)),
              'stack_quality': {
                  'kernel_nonempty': True, 'user_symbolized': 5}}
    return record, row, gate


def test_cpu_samples_filter_imports_and_keep_nested_phase_time():
    summary = phase_samples([{'count': 99, 'mask': 0}, {'count': 99, 'mask': 70}], 99)
    assert summary['training'] == {'samples': 99, 'estimated_thread_seconds': 1.0}
    assert summary['fit'] == summary['e2e'] == summary['training']
    assert summary['predict']['samples'] == 0


def test_only_requested_profile_provider_is_attached():
    cpu, offcpu = profile_program(20, 'cpu'), profile_program(20, 'offcpu')
    assert 'cpu_sample' in cpu and 'TRACEPOINT_PROBE(sched' not in cpu
    assert 'cpu_sample' not in offcpu and 'TRACEPOINT_PROBE(sched' in offcpu
    assert '== 20' in cpu and 'TARGET_CGROUP' not in cpu


def test_profile_replay_rejects_loss_and_wrong_scope():
    record, row, gate = capture()
    validate_collection(record, row, gate, b'')
    for field, value in [('lost_events', 1), ('overflow', True), ('failures', [0, 1, 0])]:
        with pytest.raises(ValueError):
            validate_collection({**record, field: value}, row, gate, b'')
    with pytest.raises(ValueError):
        validate_collection(record, row, {**gate, 'cgroup_id': 99}, b'')


def test_profile_requires_usable_symbols():
    record, row, gate = capture()
    record['stack_quality']['user_symbolized'] = 0
    with pytest.raises(ValueError, match='userspace'):
        validate_collection(record, row, gate, b'')


def test_boolean_exit_code_does_not_prove_success():
    record, row, gate = capture()
    record['stop']['exit_code'] = False
    with pytest.raises(ValueError, match='STOP'):
        validate_collection(record, row, gate, b'')


def test_one_good_frame_cannot_qualify_mostly_failed_training_stacks():
    record, row, gate = capture()
    record['samples'] = [{'count': 99, 'mask': 70, 'tgid': 42, 'user': -14, 'kernel': 2},
                         {'count': 1, 'mask': 70, 'tgid': 42, 'user': 1, 'kernel': 2}]
    record['phase_cpu_samples'] = phase_samples(record['samples'], 99)
    record['sampling_coverage'] = sampling_coverage(record['phase_cpu_samples'], row)
    row['work']['training_interval']['resources_after']['cpu']['usage_usec'] = 1010101
    record['sampling_coverage'] = sampling_coverage(record['phase_cpu_samples'], row)
    record['training_unwind_coverage'] = unwind_coverage(
        record['samples'], b'', record['stacks'], record['user_symbols'], (300, 400))
    assert record['training_unwind_coverage']['user_usable_fraction'] == .01
    with pytest.raises(ValueError, match='coverage'):
        validate_collection(record, row, gate, b'')


def test_depth_cap_is_reported_as_possible_truncation():
    samples = [{'count': 10, 'mask': 70, 'tgid': 42, 'user': 1, 'kernel': 2}]
    result = unwind_coverage(samples, b'', {'1': list(range(127)), '2': [200]},
                             {'42:1': {'symbol': 'native_function'}}, (300, 400))
    assert result['possible_truncated_fraction'] == 1


def test_sample_coverage_does_not_confuse_one_sample_with_idle_cpu():
    record, row, gate = capture()
    row['work']['training_interval']['resources_after']['cpu']['usage_usec'] = 1000000
    record['sampling_coverage'] = sampling_coverage(record['phase_cpu_samples'], row)
    with pytest.raises(ValueError, match='running-time coverage'):
        validate_collection(record, row, gate, b'')
