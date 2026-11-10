# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed accounting/clock tests independent of privileged BPF access."""

import copy
import os

import pytest
from native_io_trace import (
    DIAGNOSTICS,
    MARKERS,
    PHASES,
    mark_phase,
    prepare_marker,
    validate_trace,
)

from io_utils import fsutil


def trace_fixture():
    clients, kernels, mask = [], [], 0
    for index, (phase, entering) in enumerate(MARKERS):
        moment = 1_000_000 + index * 1000
        clients.append(dict(phase_id=phase, entering=entering, pid=91,
                            before_ns=moment, after_ns=moment + 100))
        bit = 1 << phase
        mask = (mask | bit) if entering else (mask & ~bit)
        kernels.append(dict(phase_id=phase, entering=entering, pid=91,
                            before_ns=moment + 20, after_ns=moment + 30,
                            pending_reads=0, pending_writes=0, mask=mask))
    counts = {
        name: dict(reads=10, read_bytes=40960, writes=0, write_bytes=0,
                   carried_in_reads=0, carried_out_reads=0, issued=10,
                   merged_away=0) for name in PHASES.values()
    }
    counts['total'].update(reads=40, read_bytes=163840, issued=40)
    counts['e2e'].update(reads=20, read_bytes=81920)
    return dict(pending_reads=0, pending_writes=0, request_entries=0,
                diagnostics=dict.fromkeys(DIAGNOSTICS, 0), counts=counts,
                markers=kernels), clients


def test_exact_phase_rates():
    trace, clients = trace_fixture()
    phases = validate_trace(trace, clients, 91)
    assert phases['e2e']['seconds'] == 7e-6
    assert phases['fit']['seconds'] == 3e-6
    assert phases['training']['seconds'] == 1e-6
    assert phases['training']['boundary_uncertainty_ns'] == 20
    assert phases['training']['completed_read_requests_per_second'] == 10_000_000


@pytest.mark.parametrize('field', ['pending_reads', 'pending_writes', 'request_entries'])
def test_reject_undrained(field):
    trace, clients = trace_fixture()
    trace[field] = 1
    with pytest.raises(ValueError, match='Undrained'):
        validate_trace(trace, clients, 91)


@pytest.mark.parametrize('name', DIAGNOSTICS[:-1])
def test_reject_diagnostics(name):
    trace, clients = trace_fixture()
    trace['diagnostics'][name] = 1
    with pytest.raises(ValueError, match='diagnostics'):
        validate_trace(trace, clients, 91)


def test_foreign_completions_separate():
    trace, clients = trace_fixture()
    trace['diagnostics']['untracked_device_completions'] = 1000
    assert validate_trace(trace, clients, 91)['training']['completed_read_requests'] == 10


def test_clock_offset_rejected():
    trace, clients = trace_fixture()
    trace['markers'][2]['before_ns'] += 10**9
    with pytest.raises(ValueError, match='CLOCK_MONOTONIC'):
        validate_trace(trace, clients, 91)


def test_incorrect_identity_rejected():
    trace, clients = trace_fixture()
    clients[4]['pid'] = 99
    with pytest.raises(ValueError, match='PID'):
        validate_trace(trace, clients, 91)


def test_incomplete_accounting_rejected():
    trace, clients = trace_fixture()
    trace['counts']['total']['issued'] += 1
    with pytest.raises(ValueError, match='reconcile'):
        validate_trace(trace, clients, 91)


def test_e2e_gap_is_preserved():
    trace, clients = trace_fixture()
    assert trace['markers'][4]['mask'] == (1 << 6)
    assert trace['markers'][5]['mask'] == (1 << 6) | (1 << 3)
    trace['counts']['e2e']['reads'] += 2
    trace['counts']['total']['reads'] += 2
    trace['counts']['total']['issued'] += 2
    assert validate_trace(trace, clients, 91)['e2e']['completed_read_requests'] == 22


def test_order_mask_and_negative_counts():
    trace, clients = trace_fixture()
    cases = []
    wrong = copy.deepcopy(trace)
    wrong['markers'][2]['mask'] = 0
    cases.append(wrong)
    wrong = copy.deepcopy(trace)
    wrong['markers'][2]['phase_id'] = 4
    cases.append(wrong)
    wrong = copy.deepcopy(trace)
    wrong['counts']['fit']['read_bytes'] = -1
    cases.append(wrong)
    for wrong in cases:
        with pytest.raises(ValueError):
            validate_trace(wrong, clients, 91)


@pytest.mark.skipif(os.name != 'posix' or not fsutil.exists('/proc/self/comm'),
                    reason='Linux marker syscall')
def test_marker_preserves_name():
    before = fsutil.read_text('/proc/self/comm')
    prepare_marker()
    result = mark_phase(6, True)
    assert result['before_ns'] <= result['after_ns']
    assert result['pid'] == os.getpid()
    assert fsutil.read_text('/proc/self/comm') == before
    for phase, entering in ((0, True), (7, False), (1, 1)):
        with pytest.raises(ValueError):
            mark_phase(phase, entering)
