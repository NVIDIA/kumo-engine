# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small pure cases for archive identity, exact counters, and rate aggregation."""

import copy
import json

import pytest
from native_io_iops_results import (
    derive_rates,
    digest,
    median_rates,
    seal,
    source_path,
    unseal,
    validate_envelope,
    validate_evidence,
    validate_grid,
)


def envelope_fixture():
    trace_text = '{"counts":{}}\n'
    stop_text = '{"exit_code":0,"result_path":"/old/row.json"}\n'
    row = dict(matrix_spec=dict(result='/old/row.json', iops_paths=dict(
        trace='/old/trace.json', stop='/old/stop.json')),
        iops=dict(trace='/old/trace.json', trace_sha256=digest(trace_text)))
    text = json.dumps(row)
    record = dict(source_file='/old/row.json', source_text=text,
                  source_sha256=digest(text), variant='binary-resident')
    envelope = dict(trial_source_file=record['source_file'],
                    trial_source_sha256=record['source_sha256'], variant=record['variant'],
                    trace_source_file='/old/trace.json', trace_source_text=trace_text,
                    trace_source_sha256=digest(trace_text), stop_source_file='/old/stop.json',
                    stop_source_text=stop_text, stop_source_sha256=digest(stop_text))
    return record, envelope


def test_exact_original_sidecars():
    record, envelope = envelope_fixture()
    row, trace, stop = validate_envelope(record, envelope)
    assert row['matrix_spec']['result'] == '/old/row.json'
    assert trace == {'counts': {}}
    assert stop['exit_code'] == 0


@pytest.mark.parametrize('key,value', [
    ('trial_source_file', '/other/row.json'), ('trial_source_sha256', 'a' * 64),
    ('variant', 'different'), ('trace_source_file', '/other/trace.json'),
    ('stop_source_file', '/other/stop.json'), ('trace_source_text', '{}'),
    ('stop_source_text', '{}'),
])
def test_unbound_sidecar_rejected(key, value):
    record, envelope = envelope_fixture()
    envelope[key] = value
    with pytest.raises(ValueError):
        validate_envelope(record, envelope)


@pytest.mark.parametrize('exit_code', [1, False, True, '0'])
def test_bad_stop_even_when_rehashed(exit_code):
    record, envelope = envelope_fixture()
    envelope['stop_source_text'] = json.dumps(dict(exit_code=exit_code, result_path='/old/row.json'))
    envelope['stop_source_sha256'] = digest(envelope['stop_source_text'])
    with pytest.raises(ValueError, match='STOP'):
        validate_envelope(record, envelope)


def test_altered_trace_cannot_rebind():
    record, envelope = envelope_fixture()
    envelope['trace_source_text'] = '{"counts":{"reads":999}}'
    envelope['trace_source_sha256'] = digest(envelope['trace_source_text'])
    with pytest.raises(ValueError, match='original accepted row'):
        validate_envelope(record, envelope)


def test_deterministic_archive_and_tamper(tmp_path):
    from io_utils import fsutil
    first, second = str(tmp_path / 'one.gz'), str(tmp_path / 'two.gz')
    oracle = seal(first, [dict(value=1)])
    assert seal(second, [dict(value=1)]) == oracle
    assert unseal(first, oracle) == [dict(value=1)]
    fsutil.write_binary(second, b'changed')
    with pytest.raises(ValueError, match='hash'):
        unseal(second, oracle)


def rate_fixture(reads=10, writes=6, seconds=4, native=25):
    trace = dict(counts={'training': dict(reads=reads, writes=writes,
                                          read_bytes=reads * 8192, write_bytes=writes * 4096)},
                 phases={'training': dict(seconds=seconds)})
    rates = {'training': dict(application_completed_read_sqes=None if native is None else native * 2,
                              application_read_requests_per_second=native, workload_seconds=2)}
    return trace, rates


def test_separate_denominators_and_payload():
    trace, rates = rate_fixture()
    result = derive_rates(trace, rates)['training']
    assert result['physical_read_iops'] == 2.5
    assert result['physical_write_iops'] == 1.5
    assert result['physical_total_iops'] == 4
    assert result['native_read_requests_per_second'] == 25
    assert result['mean_completed_read_payload_kib'] == 8


def test_zero_reads_not_unavailable_physical():
    trace, rates = rate_fixture(reads=0, native=None)
    result = derive_rates(trace, rates)['training']
    assert result['physical_read_iops'] == 0
    assert result['native_read_requests_per_second'] is None
    assert result['mean_completed_read_payload_kib'] is None


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf')])
def test_invalid_denominator(seconds):
    trace, rates = rate_fixture(seconds=seconds)
    with pytest.raises(ValueError, match='denominator'):
        derive_rates(trace, rates)


def test_payload_requires_completed_request():
    trace, rates = rate_fixture(reads=0)
    trace['counts']['training']['read_bytes'] = 4096
    with pytest.raises(ValueError, match='payload'):
        derive_rates(trace, rates)


def median_fixture():
    rows = []
    for repeat, (reads, writes) in enumerate([(0, 100), (0, 100), (0, 0), (100, 0), (100, 0)]):
        trace, rates = rate_fixture(reads=reads, writes=writes, seconds=1)
        metrics = derive_rates(trace, rates)['training']
        rows.append(dict(variant='v', task='binary', regime='resident', backend='direct',
                         repeat=repeat, phases=dict(training=metrics, e2e=metrics)))
    return rows


def test_total_median_is_not_sum_of_medians():
    result = median_rates(median_fixture())['v/binary/resident/direct']['training']
    assert result['physical_read_iops'] == result['physical_write_iops'] == 0
    assert result['physical_total_iops'] == 100
    assert result['mean_completed_read_payload_kib'] == 8
    assert result['payload_defined_trials'] == 2
    assert result['payload_total_trials'] == 5


def test_five_distinct_repetitions_required():
    rows = median_fixture()
    rows[-1]['repeat'] = 0
    with pytest.raises(ValueError, match='five distinct'):
        median_rates(rows)


def test_mixed_native_availability_rejected():
    rows = median_fixture()
    rows[0]['phases'] = copy.deepcopy(rows[0]['phases'])
    rows[0]['phases']['training']['native_read_requests_per_second'] = None
    with pytest.raises(ValueError, match='availability'):
        median_rates(rows)


def test_relocated_source_names():
    expected = 'packages/kumo-engine/benchmark/native_io_trace.py'
    assert source_path('/old/worktree/' + expected) == expected
    assert source_path('/different/machine/' + expected) == expected


@pytest.mark.parametrize('path', ['/old/no-source.py', 'packages/kumo-engine/../outside.py'])
def test_invalid_source_names(path):
    with pytest.raises(ValueError, match='source path'):
        source_path(path)


@pytest.mark.parametrize('native', [-1, True, 1.5])
def test_invalid_native_counts(native):
    trace, rates = rate_fixture()
    rates['training']['application_completed_read_sqes'] = native
    with pytest.raises(ValueError, match='SQE'):
        derive_rates(trace, rates)


@pytest.mark.parametrize('native_rate', [-1, float('nan'), float('inf'), True])
def test_invalid_native_rates(native_rate):
    trace, rates = rate_fixture()
    rates['training']['application_read_requests_per_second'] = native_rate
    with pytest.raises(ValueError, match='SQE'):
        derive_rates(trace, rates)


def grid_fixture():
    return [dict(task=task, regime=regime, backend=backend, repeat=repeat)
            for task in ('binary', 'regression', 'temporal_link')
            for regime in ('resident', 'constrained')
            for backend in ('mmap', 'buffered', 'direct') for repeat in range(5)]


def test_exact_full_grid():
    validate_grid(grid_fixture())


@pytest.mark.parametrize('replacement', ['duplicate', 'missing', 'one-task'])
def test_grid_substitution_rejected(replacement):
    rows = grid_fixture()
    if replacement == 'duplicate':
        rows[-1] = dict(rows[0])
    elif replacement == 'missing':
        rows.pop()
    else:
        for row in rows:
            row['task'] = 'binary'
    with pytest.raises(ValueError, match='six task/regime'):
        validate_grid(rows)


@pytest.mark.parametrize('name', ['stress20', 'public8'])
def test_missing_overhead_evidence(name):
    with pytest.raises(ValueError, match='Preserve all'):
        validate_evidence(name, [])
