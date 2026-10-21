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
        'outputs': {'predictions': 's3://out/p'},
        'warnings': ['threshold not set'],
    }
    assert 'KUMO_JOB_ID' not in os.environ


def _wrapped(cause):
    try:
        raise RuntimeError('wrapped') from cause
    except RuntimeError as error:
        return error


@pytest.mark.parametrize('error, code', [
    (_wrapped(TransientSourceError('busy')), 20),
    (ConnectionResetError(), 20),
    (TimeoutError(), 20),
    (SourceNotFoundError('missing'), 10),
    (OSError(28, 'No space left on device'), 30),
    (ValueError('x' * 2000), 30),
])
def test_failure_exits_with_a_retry_aware_code(capsys, error, code):
    with pytest.raises(SystemExit) as exited:
        with kumo.job() as run:
            run.record(snapshot_id='s-1')
            raise error
    assert exited.value.code == code
    summary = _summary(capsys)
    assert (summary['status'], summary['exit_code']) == ('failed', code)
    assert summary['snapshot_id'] == 's-1'
    assert summary['error']['type'] == type(error).__name__
    assert len(summary['error']['message']) <= 500


def test_sigterm_prints_the_summary_and_exits_143(capsys):
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as exited:
        with kumo.job():
            os.kill(os.getpid(), signal.SIGTERM)
    assert exited.value.code == 143
    assert _summary(capsys)['status'] == 'failed'
    assert signal.getsignal(signal.SIGTERM) is previous
