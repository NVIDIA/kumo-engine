# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import dataclasses
import json
from types import SimpleNamespace

import native_io_ceiling as ceiling
import pytest


def fio_report(spec):
    def job():
        options = {'rw': 'randread', 'direct': '1', 'ioengine': 'io_uring',
                   'bs': str(spec.block_bytes), 'iodepth': str(spec.depth)}
        return {'error': 0, 'job options': options,
                'read': {'runtime': 30000, 'total_ios': 30000, 'io_bytes': 30000 * spec.block_bytes,
                         'iops': 1000.0, 'bw_bytes': 1000 * spec.block_bytes, 'short_ios': 0, 'drop_ios': 0,
                         'clat_ns': {'N': 30000, 'mean': 100000, 'percentile': {'99.000000': 200000},
                                     'bins': {'100000': 30000}},
                         'slat_ns': {'mean': 1000}, 'bw_samples': 60, 'bw_mean': 4000, 'bw_dev': 5},
                'write': {'total_ios': 0, 'io_bytes': 0}, 'trim': {'total_ios': 0, 'io_bytes': 0},
                'usr_cpu': 10, 'sys_cpu': 20, 'iodepth_level': {'1': 100}}
    return {'jobs': [job() for _ in range(spec.jobs)]}


def summary_rows(high=1000, busy=50):
    return [{'block_bytes': block, 'jobs': jobs, 'depth': depth, 'submit_batch': 1, 'complete_batch': 1,
             'median_application_read_iops': high if (jobs, depth) == (4, 256) else 1000,
             'min_application_read_iops': high if (jobs, depth) == (4, 256) else 1000,
             'max_application_read_iops': high if (jobs, depth) == (4, 256) else 1000,
             'median_max_issuer_cpu_percent': busy, 'relative_mad': 0.01, 'max_trial_bandwidth_cv': .01}
            for block in (4096, 8192) for jobs, depth in ceiling.GEOMETRIES]


def test_finite_balanced_grid():
    plan = ceiling.trial_plan()
    assert len(plan) == len({spec.name for spec in plan}) == 70
    assert {(s.block_bytes, s.jobs, s.depth, s.repeat) for s in plan} == {
        (block, jobs, depth, repeat) for block in (4096, 8192)
        for jobs, depth in ceiling.GEOMETRIES for repeat in range(5)}
    assert [s.block_bytes for s in plan[:14]] != [s.block_bytes for s in plan[14:28]]


@pytest.mark.parametrize('values', [{'jobs': True}, {'repeat': 5}, {'block_bytes': 16384},
                                    {'measure_seconds': 5}, {'depth': 512}, {'submit_batch': 16}])
def test_protocol_rejects_unplanned(values):
    arguments = {'block_bytes': 4096, 'jobs': 1, 'depth': 256, 'repeat': 0} | values
    with pytest.raises(ValueError):
        ceiling.TrialSpec(**arguments)


def test_commands_are_read_only():
    spec = ceiling.TrialSpec(8192, 4, 256, 2, submit_batch=16, complete_batch=32)
    command = ceiling.fio_command(spec, '/tmp/existing.bin')
    assert '--readonly' in command and '--allow_file_create=0' in command and '--invalidate=0' in command
    assert '--rw=randread' in command and '--verify=0' in command
    assert '--iodepth_batch_submit=16' in command and '--iodepth_batch_complete_max=32' in command
    assert '--randseed=20261119' in command and '--cpus_allowed_policy=split' in command
    assert not any(argument.startswith(('--output=', '--write_', '--sqthread_poll')) for argument in command)
    crc = ceiling.crc_command('/tmp/existing.bin')
    assert '--readonly' in crc and '--verify_only=1' in crc and '--rw=read' in crc


def test_separate_job_windows_and_little_law():
    spec = ceiling.TrialSpec(4096, 4, 64, 0)
    result = ceiling.parse_fio(fio_report(spec), spec)
    assert result['application_read_iops'] == 4000
    assert result['read_bytes_per_second'] == 4000 * 4096
    assert result['estimated_application_outstanding'] == pytest.approx(0.4)
    assert 'not device' in result['outstanding_semantics']
    assert len(result['per_job']) == 4


@pytest.mark.parametrize('section,key,value', [
    ('read', 'runtime', 1000), ('read', 'total_ios', -1), ('read', 'total_ios', True),
    ('read', 'io_bytes', 4096), ('read', 'iops', 1001), ('read', 'iops', float('nan')),
    ('read', 'short_ios', 1), ('write', 'io_bytes', 4096), ('trim', 'total_ios', 1),
    ('read', 'bw_bytes', 1), ('read', 'bw_samples', 1), ('read', 'bw_dev', -1),
])
def test_fio_errors_and_incomplete_windows(section, key, value):
    spec = ceiling.TrialSpec(4096, 1, 256, 0)
    report = fio_report(spec)
    report['jobs'][0][section][key] = value
    with pytest.raises(ValueError):
        ceiling.parse_fio(report, spec)


def test_reports_preserve_all_status_objects():
    one = fio_report(ceiling.TrialSpec(4096, 1, 256, 0))
    text = json.dumps(one) + '\n\n' + json.dumps(one)
    assert ceiling.parse_reports(text) == [one, one]
    with pytest.raises(ValueError):
        ceiling.parse_reports(text + 'unexpected diagnostics')


def test_recorded_ramp_counter_boundary():
    spec = ceiling.TrialSpec(4096, 1, 16, 0)
    report = fio_report(spec)
    read = report['jobs'][0]['read']
    # Counts from the retained first live fio 3.36 QD16 qualification.
    read.update(runtime=30002, total_ios=1714442, io_bytes=7022419968,
                iops=57144.25705, bw_bytes=234065061)
    read['clat_ns'].update(N=1714457, bins={'100000': 1714457})
    result = ceiling.parse_fio(report, spec)
    counts = result['per_job'][0]['read_counts']
    assert counts['issued_reads'] == 1714442
    assert counts['completed_reads'] == 1714458
    assert counts['latency_samples'] == 1714457
    assert counts['completed_minus_issued'] == counts['ramp_boundary_bound_reads'] == 16
    assert counts['completed_minus_latency'] == 1
    assert result['application_read_iops'] == read['iops']
    assert result['estimated_application_outstanding'] == pytest.approx(1714457 / 30.002 * .0001)


@pytest.mark.parametrize('completed,latency,ramp', [
    (30017, 30000, 10), (29999, 30000, 10), (30000, 30017, 10),
    (30000, 29983, 10), (30016, 29999, 10), (30001, 30000, 0), (30000, 30001, 0),
])
def test_read_boundary_rejects_capacity_violation(completed, latency, ramp):
    read = fio_report(ceiling.TrialSpec(4096, 1, 16, 0))['jobs'][0]['read']
    read['io_bytes'] = completed * 4096
    read['clat_ns'].update(N=latency, bins={'100000': latency})
    with pytest.raises(ValueError, match='capacity'):
        ceiling.read_counts(read, 4096, 16, ramp)


def test_read_boundary_rejects_unaligned_bytes_and_bad_histogram():
    read = fio_report(ceiling.TrialSpec(4096, 1, 16, 0))['jobs'][0]['read']
    read['io_bytes'] += 1
    with pytest.raises(ValueError, match='fixed-size'):
        ceiling.read_counts(read, 4096, 16, 10)
    read['io_bytes'] -= 1
    read['clat_ns']['bins']['100000'] -= 1
    with pytest.raises(ValueError, match='histogram'):
        ceiling.read_counts(read, 4096, 16, 10)


def test_extension_is_predeclared_and_finite():
    assert ceiling.extension_blocks(summary_rows()) == []
    assert ceiling.extension_blocks(summary_rows(high=1100)) == [4096, 8192]
    assert ceiling.extension_blocks(summary_rows(busy=90)) == [4096, 8192]
    with pytest.raises(ValueError):
        ceiling.extension_blocks(summary_rows()[:-1])


def test_classification_never_promises_hardware_ceiling():
    for values, expected in [(summary_rows(), 'STORAGE_PATH_PLATEAU'),
                             (summary_rows(busy=99), 'ISSUER_BUSY'),
                             (summary_rows(high=800), 'SCALING_REGRESSION'),
                             (summary_rows(high=1200), 'THROUGHPUT_STILL_GROWING')]:
        classifications = ceiling.classify_ceiling(values)
        assert all(row['classification'] == expected for row in classifications)
        assert all(row['hardware_ceiling_proven'] is False for row in classifications)
    values = summary_rows()
    values[0]['relative_mad'] = .2
    assert ceiling.classify_ceiling(values)[0]['classification'] == 'INCONCLUSIVE_VARIABILITY'


def test_no_pooled_rates_or_lost_batch_variants():
    spec = ceiling.TrialSpec(4096, 4, 256, 0)
    rows = []
    for repeat in range(5):
        for submit, complete in [(1, 1), (16, 32)]:
            trial = dataclasses.replace(spec, repeat=repeat, submit_batch=submit, complete_batch=complete)
            rows.append({'spec': dataclasses.asdict(trial), 'metrics': {
                'application_read_iops': [100, 200, 300, 400, 10000][repeat],
                'read_bytes_per_second': 4096 * 100, 'issuer_max_cpu_percent': 50,
                'max_job_bandwidth_cv': .01, 'per_job': [{'read_counts': {
                    'issued_reads': 30000, 'completed_reads': 30000, 'latency_samples': 30000,
                    'ramp_boundary_bound_reads': 256}}]}})
    aggregate = ceiling.aggregate(rows)
    assert len(aggregate) == 2
    assert all(value['median_application_read_iops'] == 300 for value in aggregate)
    assert all([row['repeat'] for row in value['per_trial_read_counts']] == list(range(5))
               for value in aggregate)
    with pytest.raises(ValueError, match='Duplicate'):
        ceiling.aggregate(rows + [copy.deepcopy(rows[0])])


def test_execute_guard_precedes_io(monkeypatch):
    monkeypatch.setattr(ceiling, 'source_info', lambda *args: pytest.fail('No I/O before explicit execute'))
    with pytest.raises(ValueError, match='require --execute'):
        ceiling.run_sweep(argparse.Namespace(execute=False, expected_source='abc'))


def test_service_owns_limits_and_source():
    command = ceiling.service_command(ceiling.TrialSpec(4096, 1, 256, 0), '/tmp/file', 'owned.service', '/repo')
    assert 'MemoryMax=64G' in command and 'MemorySwapMax=0' in command
    assert 'KillMode=control-group' in command and '--execute' in command
    assert 'WorkingDirectory=/repo' in command and 'PYTHONPATH=/repo' in command
    assert '--uid=' + str(ceiling.os.getuid()) in command


def test_thermal_gate():
    clean = {'critical_warning': 0, 'media_errors': 0, 'thm_temp1_trans_count': 0, 'thm_temp2_trans_count': 0}
    ceiling.check_thermal(clean, dict(clean))
    with pytest.raises(ValueError):
        ceiling.check_thermal(clean, clean | {'thm_temp1_trans_count': 1})


def test_cleanup_runs_after_failed_service(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if 'systemd-run' in command:
            return SimpleNamespace(returncode=1, stdout='raw failure output', stderr='failure')
        if 'is-active' in command:
            return SimpleNamespace(returncode=3, stdout='inactive', stderr='')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    clean = {'critical_warning': 0, 'media_errors': 0, 'thm_temp1_trans_count': 0, 'thm_temp2_trans_count': 0}
    monkeypatch.setattr(ceiling.subprocess, 'run', run)
    monkeypatch.setattr(ceiling, 'thermal_info', lambda: dict(clean))
    row = ceiling.run_trial(ceiling.TrialSpec(4096, 1, 256, 0), '/tmp/file', '/repo')
    assert row['status'] == 'FAIL' and row['service_stdout'] == 'raw failure output'
    assert any('stop' in command for command in calls)
    assert any('is-active' in command for command in calls)


def test_failed_trial_saved_before_abort(monkeypatch):
    saved = []
    monkeypatch.setattr(ceiling, 'source_info', lambda *args: {'head': 'pin'})
    monkeypatch.setattr(ceiling, 'run_trial', lambda *args: {'status': 'FAIL', 'raw': 'preserved'})
    monkeypatch.setattr(ceiling, 'save', lambda path, value: saved.append((path, value)))
    args = argparse.Namespace(repo_root='/repo', expected_source='pin', fixture='/tmp/file', output='/tmp/out')
    with pytest.raises(RuntimeError, match='diagnostics saved'):
        ceiling.accept_trial(ceiling.TrialSpec(4096, 1, 256, 0), args, {'head': 'pin'})
    assert saved[0][1]['raw'] == 'preserved'


def test_timeout_preserves_fio_output_and_samples(monkeypatch):
    class Process:
        pid = 123
        returncode = None
        calls = 0

        def communicate(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise ceiling.subprocess.TimeoutExpired('fio', timeout, output=b'partial')
            self.returncode = -9
            return 'all partial fio stdout', 'fio diagnostic'

        def poll(self):
            return self.returncode

    class Observer:
        def __init__(self, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(ceiling.os, 'sched_setaffinity', lambda *args: None)
    monkeypatch.setattr(ceiling.os, 'killpg', lambda *args: None)
    monkeypatch.setattr(ceiling, 'cgroup_info', lambda: {'path': '/fake/cgroup'})
    monkeypatch.setattr(ceiling, 'fixture_info', lambda path: {'path': path, 'inode': 1})
    monkeypatch.setattr(ceiling, 'trial_snapshot', lambda path: {'sample': 1})
    monkeypatch.setattr(ceiling.fsutil, 'read_text', lambda path: 'oom 0\noom_kill 0\n')
    monkeypatch.setattr(ceiling.threading, 'Thread', Observer)
    monkeypatch.setattr(ceiling.subprocess, 'Popen', lambda *args, **kwargs: Process())
    row = ceiling.execute_child(ceiling.TrialSpec(4096, 1, 256, 0), '/tmp/file')
    assert row['status'] == 'FAIL'
    assert row['fio_stdout'] == 'all partial fio stdout' and row['fio_stderr'] == 'fio diagnostic'
    assert row['kernel_samples'] == [{'sample': 1}]
    assert 'TimeoutExpired' in row['execution_error']
