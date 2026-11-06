# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure detection in the standalone runner; no product qualification."""

import importlib.util
import json
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


@pytest.mark.parametrize(
    'fault',
    ['changed_core', 'foreign_core', 'changed_native', 'changed_ingest'])
def test_installed_components_must_be_supplied_by_exact_wheel(
        tmp_path, monkeypatch, fault):
    from types import ModuleType

    prefix = tmp_path / 'venv'
    location = prefix / 'site-packages'
    prefix.mkdir()
    (prefix / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    names = {
        'kumo.engine': 'kumo/engine/__init__.py',
        'kumo_core': 'kumo_core/__init__.py',
        'diskgraph': 'diskgraph/__init__.py',
        'diskgraph._c': 'diskgraph/_c.abi3.so'
    }
    modules = {}
    files = list(names.values()) + ['diskgraph/bin/ingest']
    wheel = tmp_path / 'candidate.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name in files:
            path = location / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            archive.write(path, name)
    for name, relative in names.items():
        module = ModuleType(name)
        module.__file__ = str(location / relative)
        modules[name] = module
    modules['kumo.engine'].__version__ = '1'
    parent = ModuleType('kumo')
    parent.engine = modules['kumo.engine']
    modules['kumo'] = parent
    modules['diskgraph']._c = modules['diskgraph._c']
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    direct = {'archive_info': {'hashes': {'sha256': runner.digest(wheel)}}}
    dist = SimpleNamespace(version='1', files=files,
                           read_text=lambda _: json.dumps(direct),
                           locate_file=lambda name: location / name)
    monkeypatch.setattr(runner.importlib.metadata, 'distribution',
                        lambda _: dist)
    monkeypatch.setattr(runner.sys, 'prefix', str(prefix))
    runner.check_install(wheel, '1')
    if fault == 'foreign_core':
        dist.files = [name for name in files if name != names['kumo_core']]
        message = 'not supplied by the candidate wheel'
    else:
        target = {
            'changed_core': names['kumo_core'],
            'changed_native': names['diskgraph._c'],
            'changed_ingest': 'diskgraph/bin/ingest'
        }[fault]
        (location / target).write_bytes(b'other installed build')
        message = 'differs from the selected wheel'
    with pytest.raises(AssertionError, match=message):
        runner.check_install(wheel, '1')


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
