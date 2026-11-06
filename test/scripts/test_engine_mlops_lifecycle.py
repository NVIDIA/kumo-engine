# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure detection in the standalone runner; no product qualification."""

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / 'packages' / 'kumo-engine' /
          'test' / 'wheel_mlops_lifecycle.py')
SPEC = importlib.util.spec_from_file_location('mlops_lifecycle', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def output():
    p = np.linspace(0.1, 0.9, runner.POPULATION)
    return SimpleNamespace(
        frame=pd.DataFrame({
            'ENTITY': range(runner.POPULATION),
            'TARGET_PRED': p > 0.5,
            'False_PROB': 1 - p,
            'True_PROB': p
        }), requested_rows=runner.POPULATION, scored_rows=runner.POPULATION,
        skipped_rows=0, manifest=SimpleNamespace(
            prediction_table=SimpleNamespace(anchors=['2024-03-30'])))


def test_population_checks_default_schema_without_an_anchor_column():
    result = runner.population(output())
    assert list(result) == runner.SCHEMA


@pytest.mark.parametrize('fault', [
    'missing', 'duplicate', 'foreign', 'nan', 'range', 'sum', 'label',
    'accounting', 'schema', 'anchor'
])
def test_invalid_predictions_cannot_qualify(fault):
    run = output()
    if fault == 'missing':
        run.frame = run.frame.iloc[:-1]
    elif fault == 'duplicate':
        run.frame.loc[1, 'ENTITY'] = 0
    elif fault == 'foreign':
        run.frame.loc[1, 'ENTITY'] = 10**9
    elif fault == 'nan':
        run.frame.loc[1, 'True_PROB'] = np.nan
    elif fault == 'range':
        run.frame.loc[1, 'True_PROB'] = 1.1
    elif fault == 'sum':
        run.frame.loc[1, 'False_PROB'] = 0.0
    elif fault == 'label':
        run.frame.loc[1, 'TARGET_PRED'] = True
    elif fault == 'accounting':
        run.skipped_rows = 1
    elif fault == 'schema':
        run.frame['BUNDLE_ID'] = 'unexpected'
    else:
        run.manifest.prediction_table.anchors = []
    with pytest.raises(AssertionError):
        runner.population(run)


def test_wrong_resolved_model_cannot_pass_tracking_checks():
    tags = {
        'kumo.model_name': runner.MODEL,
        'kumo.model_version': '2',
        'kumo.bundle_id': 'bundle',
        'kumo.prediction_id': 'prediction'
    }
    record = SimpleNamespace(info=SimpleNamespace(status='FINISHED'),
                             data=SimpleNamespace(tags=tags))
    from contextlib import nullcontext
    cfg = SimpleNamespace(tracking=SimpleNamespace(
        client=lambda: nullcontext(SimpleNamespace(get_run=lambda _: record))))
    run = SimpleNamespace(id='prediction',
                          tracking=SimpleNamespace(run_id='run'))
    expected = {'version': '1', 'bundle': 'bundle'}
    with pytest.raises(AssertionError, match='wrong model/run'):
        runner.prediction_tracking(cfg, run, expected)


def test_preflight_failure_retains_a_blocked_receipt(tmp_path, monkeypatch):
    def unavailable(*args):
        raise AssertionError('Wrong installed candidate wheel')

    monkeypatch.setattr(runner, 'check_install', unavailable)
    args = SimpleNamespace(output_dir=tmp_path / 'attempt',
                           wheel=tmp_path / 'x', expected_version='1',
                           phase_timeout=1)
    assert runner.run(args) == 1
    summary = runner.read(args.output_dir / 'summary.json')
    assert summary['status'] == 'blocked'
    assert 'Wrong installed candidate wheel' in summary['error']
    assert all(p['status'] == 'blocked' for p in summary['phases'].values())
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_phase_timeout_is_bounded(tmp_path):
    with (tmp_path / 'worker.log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            runner.execute(
                [sys.executable, '-c', 'import time; time.sleep(60)'], tmp_path,
                log, 0.1)


def test_worker_access_blocker_stays_blocked(tmp_path, monkeypatch):
    root = tmp_path / 'attempt'
    monkeypatch.setattr(runner, 'check_install', lambda *args: {})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: None)

    def blocked(command, cwd, log, timeout):
        runner.write(cwd / 'fit.json', {
            'status': 'blocked',
            'error': 'Optional tracking package missing'
        })
        return 1

    monkeypatch.setattr(runner, 'execute', blocked)
    args = SimpleNamespace(output_dir=root, wheel=tmp_path / 'x',
                           expected_version='1', phase_timeout=1)
    assert runner.run(args) == 1
    summary = runner.read(root / 'summary.json')
    assert summary['status'] == summary['phases']['fit']['status'] == 'blocked'
    assert summary['phases']['fit']['returncode'] == 1
    assert summary['phases']['score']['status'] == 'blocked'


def test_installed_native_bytes_must_match_candidate_archive(tmp_path):
    location = tmp_path / 'site-packages'
    native = location / 'diskgraph' / '_c.abi3.so'
    native.parent.mkdir(parents=True)
    native.write_bytes(b'selected native build')
    wheel = tmp_path / 'candidate.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.write(native, 'diskgraph/_c.abi3.so')
    runner.verify_wheel_file(wheel, location, native)
    native.write_bytes(b'stale native build')
    with pytest.raises(AssertionError, match='differs from the selected wheel'):
        runner.verify_wheel_file(wheel, location, native)


def test_internal_platform_entry_requires_explicit_execution(tmp_path):
    with pytest.raises(SystemExit) as error:
        runner.main([
            '--platform-config',
            str(tmp_path / 'profile.json'), '--wheel', 'unused.whl',
            '--expected-version', '1', '--output-dir',
            str(tmp_path / 'attempt')
        ])
    assert error.value.code == 2


def test_cleanup_failure_is_failed_not_blocked(tmp_path, monkeypatch):
    root = tmp_path / 'attempt'
    monkeypatch.setattr(runner, 'check_install', lambda *args: {})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: None)

    def completed(command, cwd, log, timeout):
        (cwd / 'scratch-fit').mkdir()
        runner.write(cwd / 'fit.json', {'status': 'passed'})
        return 0

    def cannot_remove(path):
        raise OSError('Cannot remove producer scratch')

    monkeypatch.setattr(runner, 'execute', completed)
    monkeypatch.setattr(runner.shutil, 'rmtree', cannot_remove)
    args = SimpleNamespace(output_dir=root, wheel=tmp_path / 'x',
                           expected_version='1', phase_timeout=1)
    assert runner.run(args) == 1
    report = runner.read(root / 'summary.json')
    assert report['status'] == 'failed'
    assert 'Cannot remove producer scratch' in report['error']
    assert report['phases']['score']['status'] == 'blocked'


def test_platform_permission_prerequisite_runs_before_any_fit(
        tmp_path, monkeypatch):
    root = tmp_path / 'attempt'
    profile = SimpleNamespace(config=dict(
        environment='managed', execution_context='snowflake_notebook'))
    monkeypatch.setattr(runner, 'platform_backend', lambda _: profile)
    monkeypatch.setattr(runner, 'check_install', lambda *args, **kwargs: {})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: None)
    phases = []

    def blocked(command, cwd, log, timeout):
        phase = command[command.index('--phase') + 1]
        phases.append(phase)
        runner.write(cwd / (phase + '.json'),
                     dict(status='blocked', error='Denied stage not ready'))
        return 1

    monkeypatch.setattr(runner, 'execute', blocked)
    args = SimpleNamespace(output_dir=root, wheel=tmp_path / 'x',
                           expected_version='1', phase_timeout=1)
    assert runner.run(args) == 1
    report = runner.read(root / 'summary.json')
    assert phases == ['negative']
    assert report['phases']['negative']['status'] == 'blocked'
    assert report['phases']['fit']['status'] == 'blocked'
    assert report['required_gaps']['exported_epoch_history'][
        'status'] == 'not implemented'
