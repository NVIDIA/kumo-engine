# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import warnings

import pytest
from kumo_core.sources.errors import SourceNotFoundError, TransientSourceError

import kumo.engine as kumo
from kumo.engine._manifest import new_manifest


@pytest.fixture(autouse=True)
def no_scheduler(monkeypatch):
    for name in ('KUMO_SCHEDULER', 'KUMO_JOB_ID', 'KUMO_JOB_RUN_ID'):
        monkeypatch.delenv(name, raising=False)


def _summary(capsys):
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    return json.loads(out[0])


def test_summary_records_ids_warnings_and_scheduler(capsys):
    with kumo.job(job_id='job-7', job_run_id='run-7') as run:
        manifest = new_manifest('model_bundle', 'b-1', {})
        run.record(bundle_id='b-1', snapshot_id='s-1')
        run.record(run_id='r-1', outputs={'predictions': 's3://out/p'})
        warnings.warn('threshold not set', UserWarning)
        warnings.warn('use the new API', FutureWarning)
        for i in range(30):
            warnings.warn(f'{i}' * 400, RuntimeWarning)
            warnings.warn(f'{i}' * 400, RuntimeWarning)

    assert manifest['scheduler'] == {
        'platform': 'other',
        'job_id': 'job-7',
        'job_run_id': 'run-7'
    }
    summary = _summary(capsys)
    assert summary == {
        'event': 'kumo.engine.job',
        'status': 'succeeded',
        'exit_code': 0,
        'scheduler': manifest['scheduler'],
        'bundle_id': 'b-1',
        'snapshot_id': 's-1',
        'run_id': 'r-1',
        'outputs': {
            'predictions': 's3://out/p'
        },
        'warnings': summary['warnings'],
    }
    assert summary['warnings'][0] == 'threshold not set'
    assert 'use the new API' not in summary['warnings']
    assert len(summary['warnings']) == 20
    assert len(set(summary['warnings'])) == 20
    assert max(map(len, summary['warnings'])) == 300
    assert 'KUMO_JOB_ID' not in os.environ


def test_warning_survives_a_patcher_that_wraps_kumos_showwarning(capsys):
    original = {}

    def patched(*args, **kwargs):
        original['show'](*args, **kwargs)

    with pytest.warns(UserWarning, match='threshold not set'):
        original['show'] = warnings.showwarning
        warnings.showwarning = patched
        with kumo.job():
            original['show'] = warnings.showwarning
            warnings.showwarning = patched
            warnings.warn('threshold not set', UserWarning)

    assert _summary(capsys)['warnings'] == ['threshold not set']


def _wrapped(cause):
    try:
        raise RuntimeError('wrapped') from cause
    except RuntimeError as error:
        return error


@pytest.mark.parametrize('error, kind', [
    (_wrapped(TransientSourceError('busy')), 'transient'),
    (ConnectionResetError(), 'transient'),
    (SourceNotFoundError('missing'), 'permanent'),
    (ValueError('x' * 2000), 'unknown'),
])
def test_failure_exits_non_zero_with_a_retry_hint(capsys, error, kind):
    with pytest.raises(SystemExit) as exited:
        with kumo.job(job_id='job-7') as run:
            run.record(snapshot_id='s-1')
            raise error
    assert exited.value.code == 1
    summary = _summary(capsys)
    assert (summary['status'], summary['exit_code']) == ('failed', 1)
    assert summary['snapshot_id'] == 's-1'
    assert summary['scheduler']['job_id'] == 'job-7'
    assert summary['error']['type'] == type(error).__name__
    assert summary['error']['kind'] == kind
    assert len(summary['error']['message']) <= 500
    assert 'KUMO_JOB_ID' not in os.environ


@pytest.mark.parametrize('error, code', [
    (KeyboardInterrupt(), 130),
    (SystemExit(3), 3),
    (SystemExit(0), 0),
])
def test_interrupt_and_exit_still_print_the_summary(capsys, error, code):
    with pytest.raises(type(error)):
        with kumo.job(job_id='job-7'):
            raise error
    summary = _summary(capsys)
    assert summary['exit_code'] == code
    assert summary['status'] == ('succeeded' if code == 0 else 'failed')
    assert 'KUMO_JOB_ID' not in os.environ


def test_sigterm_prints_the_summary_and_exits_143(capsys):
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as exited:
        with kumo.job():
            os.kill(os.getpid(), signal.SIGTERM)
    assert exited.value.code == 143
    assert _summary(capsys)['status'] == 'failed'
    assert signal.getsignal(signal.SIGTERM) is previous
