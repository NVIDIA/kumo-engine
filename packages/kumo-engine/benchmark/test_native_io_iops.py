# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check service isolation, marker boundaries and attribution failure gates."""

import argparse
import copy
import os
import time

import native_io_iops
import pytest
from native_io_iops import (
    new_spec,
    phase_rates,
    service_command,
    stop_collector,
    stop_service,
    summarize_rates,
    validate_args,
    validate_attribution,
    validate_ready,
    validate_result,
    validate_resume,
)
from native_io_matrix import training_evidence
from native_io_phases import (
    MARKER_ORDER,
    PHASE_IDS,
    PhaseRecorder,
    parse_device_io,
)


def arguments(tmp_path):
    return argparse.Namespace(
        output_dir=str(tmp_path), trace_mode='on', phase_markers='1', device=None,
        service_user='dmawhirter', request_capacity=8192, drain_seconds=10.,
        startup_timeout=180., collector_cpus='0-4,10-14', pilot=True,
        steps=None, python=None, build_provenance=None, repetitions=2,
        expected_native_sha256='a' * 64)


def template():
    return {'matrix_spec': dict(
        task='binary', regime='constrained', backend='direct', workers=0,
        threads=4, torch_threads=1, loader_mode='standalone', memory_bytes=4 * 1024**3,
        steps=1280, batch_size=1280, cpus='5-9,15-19', timeout=1800,
        direct_cache_gib=2, io_rings=8, io_qd=256, io_batch_mult=1, model_seed=0,
        python='/tmp/pinned-python', runner='/tmp/runner.py', prepared_dir='/tmp/data',
        prepared_manifest_sha256='b' * 64, build_provenance_sha256='c' * 64)}


def test_service_preserves_controls(tmp_path):
    original = template()
    spec = new_spec(original, arguments(tmp_path), 1, {'git_sha': 'new-source'})
    command = service_command(spec, 'dg-native-iops-' + 'a' * 32 + '.service', 'dmawhirter')
    assert command[:4] == ['sudo', '-n', 'systemd-run', '--quiet']
    assert '--user' not in command
    assert '--uid=dmawhirter' in command
    assert f'MemoryMax={4 * 1024**3}' in command
    assert 'MemorySwapMax=0' in command
    assert 'IOAccounting=yes' in command
    assert command[command.index('taskset') + 2] == original['matrix_spec']['cpus']
    for key in ('steps', 'batch_size', 'prepared_dir', 'prepared_manifest_sha256',
                'direct_cache_gib', 'io_rings', 'io_qd', 'io_batch_mult'):
        assert spec[key] == original['matrix_spec'][key]
    assert spec['iops_environment']['OMP_NUM_THREADS'] == '1'
    assert spec['iops_environment']['DG_IOPS_PHASE_MARKERS'] == '1'
    assert spec['source_provenance']['git_sha'] == 'new-source'
    assert spec['result'].startswith(str(tmp_path))
    assert spec['runtime_prepared_dir'] != original['matrix_spec']['prepared_dir']
    assert 'iops_config' not in original['matrix_spec']


@pytest.mark.parametrize('field,value', [('workers', 1), ('torch_threads', 4),
                                         ('memory_bytes', 2 * 1024**3)])
def test_reject_changed_work_controls(tmp_path, field, value):
    row = template()
    row['matrix_spec'][field] = value
    with pytest.raises(ValueError):
        new_spec(row, arguments(tmp_path), 0, {})


def test_gate_rejects_other_service():
    gate = dict(pid=42, cgroup_id=100)
    validate_ready(dict(status='READY', gate_pid=42, cgroup_id=100), gate)
    for field, value in [('gate_pid', 43), ('cgroup_id', 101), ('status', 'FAILED')]:
        ready = dict(status='READY', gate_pid=42, cgroup_id=100)
        ready[field] = value
        with pytest.raises(RuntimeError):
            validate_ready(ready, gate)
    with pytest.raises(ValueError, match='unrelated'):
        stop_service('someone-elses.service')


def test_marker_default_off(monkeypatch):
    monkeypatch.delenv('DG_IOPS_PHASE_MARKERS', raising=False)
    recorder = PhaseRecorder.from_environment()
    assert recorder.marker is None and recorder.io_path is None
    monkeypatch.setenv('DG_IOPS_PHASE_MARKERS', 'yes')
    with pytest.raises(ValueError):
        PhaseRecorder.from_environment()


def test_nested_intervals_and_markers():
    def marker(phase, entering):
        before = time.perf_counter_ns()
        return dict(phase_id=phase, entering=int(entering), pid=os.getpid(),
                    before_ns=before, after_ns=time.perf_counter_ns())

    recorder = PhaseRecorder(marker)
    names = {value: key for key, value in PHASE_IDS.items()}
    for phase, entering in MARKER_ORDER:
        (recorder.begin if entering else recorder.end)(names[phase])
    recorder.validate()
    fit, training, e2e = (recorder.intervals[key] for key in ('fit', 'training', 'e2e'))
    assert e2e['start_ns'] < fit['start_ns'] < training['start_ns']
    assert training['end_ns'] < fit['end_ns'] < e2e['end_ns']
    assert recorder.intervals['predict']['end_ns'] < e2e['end_ns']
    assert [(item['phase_id'], item['entering']) for item in recorder.markers] == MARKER_ORDER


def test_reject_incomplete_intervals():
    recorder = PhaseRecorder()
    with pytest.raises(ValueError, match='inside fit'):
        recorder.begin('training')
    recorder.begin('fit')
    with pytest.raises(ValueError, match='Incomplete'):
        recorder.validate()
    with pytest.raises(ValueError, match='repeated'):
        recorder.begin('fit')


def test_device_counts_not_summed():
    devices = parse_device_io('259:0 rios=4 rbytes=16384\n259:2 rios=4 rbytes=16384\n')
    assert devices['259:0']['rios'] == 4
    assert devices['259:2']['rios'] == 4
    with pytest.raises(ValueError, match='Duplicate'):
        parse_device_io('259:0 rios=1\n259:0 rios=2')


def test_reject_silent_attribution_failure():
    trace = dict(device=dict(io_stat_key='259:0'),
                 counts=dict(total=dict(reads=0), training=dict(reads=0)))
    row = dict(regime='constrained', training_evidence=dict(
        has_pressure_evidence=True, pid_physical_read_bytes=4096),
               phase_io_devices=dict(training=dict(
                   before=dict(devices={'259:0': dict(rios=1, rbytes=4096)}),
                   after=dict(devices={'259:0': dict(rios=8, rbytes=32768)}))))
    with pytest.raises(ValueError, match='no attributed completions'):
        validate_attribution(trace, row)
    trace['counts'] = dict(total=dict(reads=9), training=dict(reads=7))
    validate_attribution(trace, row)
    row['phase_io_devices']['training']['after']['devices'] = {}
    with pytest.raises(ValueError, match='target-device cgroup'):
        validate_attribution(trace, row)


def test_resume_rejects_status_change(tmp_path):
    spec = new_spec(template(), arguments(tmp_path), 0, {})
    row = dict(iops=dict(config=copy.deepcopy(spec['iops_config']), status='FAILED'))
    with pytest.raises(ValueError, match='status'):
        validate_resume(spec, row)


@pytest.mark.parametrize('field,value', [('steps', 0), ('repetitions', 0),
                                         ('request_capacity', 0), ('drain_seconds', 0),
                                         ('startup_timeout', 0)])
def test_reject_invalid_run_bounds(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match='positive'):
        validate_args(args)


def test_keep_venv_executable_identity(tmp_path):
    executable = tmp_path / 'python'
    executable.symlink_to('/usr/bin/python3')
    args = arguments(tmp_path)
    args.python = str(executable)
    spec = new_spec(template(), args, 0, {})
    assert spec['python'] == str(executable)


def test_formal_keeps_original_steps(tmp_path):
    args = arguments(tmp_path)
    args.pilot, args.steps = False, 256
    with pytest.raises(ValueError, match='step count'):
        new_spec(template(), args, 0, {})


def test_rate_denominators_stay_separate():
    row = dict(backend='direct', phase_seconds=dict(fit=2.), phase_intervals_ns=dict(fit=dict(
        start_ns=1_000_000_000, end_ns=3_000_000_000)), native_io=dict(fit=dict(
            before=dict(io_ops=10), after=dict(io_ops=30))))
    trace = dict(markers=[dict(phase_id=1, after_ns=900_000_000),
                          dict(phase_id=1, before_ns=3_100_000_000, after_ns=3_100_000_000)],
                 phases=dict(fit=dict(seconds=2.2, completed_read_requests=11,
                                      completed_read_requests_per_second=5.)))
    rates = phase_rates(row, trace)['fit']
    assert rates['application_read_requests_per_second'] == 10
    assert rates['physical']['seconds'] == 2.2
    assert rates['physical']['completed_read_requests_per_second'] == 5
    assert rates['physical']['workload_start_after_kernel_ns'] == 100_000_000
    row['backend'] = 'mmap'
    assert phase_rates(row, trace)['fit']['application_read_requests_per_second'] is None


def test_median_of_per_trial_rates():
    rows = [dict(backend='buffered', iops=dict(phase_rates=dict(training=dict(
        application_read_requests_per_second=count / seconds, physical=None))))
            for count, seconds in [(5, 1), (10, 100), (100, 10)]]
    summary = summarize_rates(rows)['buffered']['phase_rate_medians']['training']
    assert summary['application_read_requests_per_second'] == 5


def eligibility_row(seconds=60., scan=1):
    def snapshot(read_bytes, pgscan):
        return dict(observer_process=dict(pid=4, io=dict(read_bytes=read_bytes)),
                    memory_stat=dict(pgscan=pgscan))

    return dict(provenance=dict(native_sha256='a' * 64, index_bytes=100), work=dict(
        training_interval=dict(seconds=seconds, resources_before=snapshot(0, 0),
                               resources_after=snapshot(4096, scan),
                               native_before=dict(evictions=0), native_after=dict(evictions=0))))


@pytest.mark.parametrize('seconds,scan,error', [(10., 1, 'short'), (60., 0, 'pressure')])
def test_replay_rechecks_formal_eligibility(seconds, scan, error):
    row = eligibility_row(seconds, scan)
    spec = dict(expected_native_sha256='a' * 64, workers=0, memory_bytes=40,
                regime='constrained', pilot=False, min_training_seconds=30)
    row['training_evidence'] = training_evidence(row, spec)
    with pytest.raises(ValueError, match=error):
        validate_result(row, spec)


def test_fresh_native_pin_required():
    row = dict(provenance=dict(native_sha256='b' * 64))
    with pytest.raises(ValueError, match='native library'):
        validate_result(row, dict(expected_native_sha256='a' * 64))


def test_collector_stops_after_service_cleanup_failure(tmp_path, monkeypatch):
    spec = new_spec(template(), arguments(tmp_path), 0, {})
    gate = dict(pid=42, cgroup_id=100)
    finished = []

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            finished.append(self)
            return 0

    processes = [Process(), Process()]
    pending = iter(processes)
    monkeypatch.setattr(native_io_iops.subprocess, 'Popen', lambda *args, **kwargs: next(pending))
    monkeypatch.setattr(native_io_iops, 'wait_json', lambda path, process, timeout: (
        dict(status='READY', gate_pid=42, cgroup_id=100) if path == spec['iops_paths']['ready']
        else gate))

    def fail_stop(unit):
        raise RuntimeError('service cleanup failure')

    monkeypatch.setattr(native_io_iops, 'stop_service', fail_stop)
    with pytest.raises(RuntimeError, match='service cleanup failure'):
        native_io_iops.run_service(spec, 'dmawhirter')
    assert processes[1] in finished
    assert native_io_iops.read_json(spec['iops_paths']['stop'])['exit_code'] == 0


def test_wedged_collector_is_killed(monkeypatch):
    calls = []

    class Process:
        pid = 87654
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits <= 2:
                raise native_io_iops.subprocess.TimeoutExpired('owned-collector', timeout)
            return -9

        def send_signal(self, signal):
            calls.append(signal)

    monkeypatch.setattr(native_io_iops.subprocess, 'run',
                        lambda command, **kwargs: calls.append(command))
    with pytest.raises(RuntimeError, match='forced termination'):
        stop_collector(Process(), 10)
    assert calls == [2, ['sudo', '-n', 'kill', '-KILL', '--', '-87654']]


def test_wedged_owned_service_is_killed(monkeypatch):
    calls = []
    unit = 'dg-native-iops-' + 'a' * 32 + '.service'

    def command(args, **kwargs):
        calls.append(args)
        if 'stop' in args:
            raise native_io_iops.subprocess.TimeoutExpired(args, 30)
        return native_io_iops.subprocess.CompletedProcess(args, 0, stdout='inactive')

    monkeypatch.setattr(native_io_iops.subprocess, 'run', command)
    stop_service(unit)
    assert calls[1] == ['sudo', '-n', 'systemctl', 'kill', '--kill-whom=all',
                        '--signal=KILL', unit]
