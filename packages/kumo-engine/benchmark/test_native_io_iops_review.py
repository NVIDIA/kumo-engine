# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent replay tamper tests for physical request attribution."""

import copy

import pytest
from native_io_trace import (
    DIAGNOSTICS,
    MARKERS,
    PHASES,
    validate_collection,
    validate_trace,
)


def collection_fixture():
    clients, markers, mask = [], [], 0
    for index, (phase, entering) in enumerate(MARKERS):
        moment = 100_000 + index * 1000
        clients.append(dict(phase_id=phase, entering=entering, pid=71,
                            before_ns=moment, after_ns=moment + 100))
        mask = mask | (1 << phase) if entering else mask & ~(1 << phase)
        markers.append(dict(phase_id=phase, entering=entering, pid=71,
                            before_ns=moment + 20, after_ns=moment + 40,
                            pending_reads=0, pending_writes=0, mask=mask))
    counts = {name: dict(reads=4, read_bytes=16384, writes=0, write_bytes=0,
                         carried_in_reads=0, carried_out_reads=0, issued=4,
                         merged_away=0, root_issued_reads=0, root_issued_writes=0,
                         requeues=0, reissues=0, partials=0)
              for name in PHASES.values()}
    for name, reads in {'total': 32, 'fit': 8, 'e2e': 16}.items():
        counts[name].update(reads=reads, read_bytes=reads * 4096, issued=reads)
    gate = dict(pid=70, cgroup_id=99, prepared_dir='/tmp/owned',
                cgroup_path='/sys/fs/cgroup/system.slice/owned.service')
    identity = dict(cgroup_id=99, device=dict(major=259, minor=0,
                                              io_stat_key='259:0'),
                    collector_sha256='a' * 64, bpf_source_sha256='b' * 64,
                    btf_sha256='c' * 64, request_capacity=8192, marker_capacity=16)
    trace = dict(status='PASS', gate=gate, ready=dict(status='READY', **identity),
                 counts=counts, markers=markers, diagnostics=dict.fromkeys(DIAGNOSTICS, 0),
                 pending_reads=0, pending_writes=0, request_entries=0, **identity)
    trace['phases'] = validate_trace(trace, clients, 71)
    result = dict(iops_phase_markers=clients, native_io=dict(fit=dict(after=dict(pid=71))))
    return trace, result, gate


def test_replay_recomputes_rates():
    trace, result, gate = collection_fixture()
    validate_collection(trace, result, gate)
    trace['phases']['training']['completed_read_requests_per_second'] += 1
    with pytest.raises(ValueError, match='raw counters'):
        validate_collection(trace, result, gate)


@pytest.mark.parametrize('field', ['cgroup_id', 'collector_sha256', 'bpf_source_sha256',
                                   'btf_sha256', 'request_capacity', 'marker_capacity'])
def test_replay_rejects_identity_change(field):
    trace, result, gate = collection_fixture()
    trace['ready'][field] = 0
    with pytest.raises(ValueError, match='identity'):
        validate_collection(trace, result, gate)


def test_replay_rejects_device_alias():
    trace, result, gate = collection_fixture()
    trace['device']['io_stat_key'] = '259:2'
    with pytest.raises(ValueError, match='Device accounting'):
        validate_collection(trace, result, gate)


@pytest.mark.parametrize('field', ['reads', 'read_bytes', 'writes', 'write_bytes'])
def test_reject_impossible_nested_counts(field):
    trace, result, _ = collection_fixture()
    trace['counts']['training'][field] = trace['counts']['fit'][field] + 1
    with pytest.raises(ValueError):
        validate_trace(trace, result['iops_phase_markers'], 71)


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_reject_non_count_values(value):
    trace, result, _ = collection_fixture()
    trace['counts']['training']['reads'] = value
    with pytest.raises(ValueError, match='counter'):
        validate_trace(trace, result['iops_phase_markers'], 71)


def test_failed_status_not_rescued():
    trace, result, gate = collection_fixture()
    trace['status'] = 'FAIL'
    with pytest.raises(ValueError, match='Failed collector'):
        validate_collection(trace, result, gate)


def test_kernel_markers_cannot_impersonate_client():
    trace, result, gate = collection_fixture()
    wrong = copy.deepcopy(result)
    wrong['iops_phase_markers'][0]['pid'] = 72
    with pytest.raises(ValueError, match='PID'):
        validate_collection(trace, wrong, gate)
