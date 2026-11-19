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


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('fault', [
    'changed_core', 'foreign_core', 'changed_native', 'changed_ingest',
    'wrong_hash'
])
def test_installed_components_must_be_supplied_by_exact_wheel(
        tmp_path, monkeypatch, fault, managed):
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
    runner.check_install(wheel, '1', managed=managed)
    if fault == 'wrong_hash':
        direct['archive_info']['hashes']['sha256'] = '0' * 64
        message = 'Install the exact candidate wheel with pip'
    elif fault == 'foreign_core':
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
        runner.check_install(wheel, '1', managed=managed)


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


@pytest.mark.parametrize('ready', [False, True])
def test_platform_permission_prerequisite_runs_before_any_fit(
        tmp_path, monkeypatch, ready):
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
        runner.write(
            cwd / (phase + '.json'),
            dict(status='passed') if ready else dict(
                status='blocked', error='Denied stage not ready'))
        return 0 if ready else 1

    monkeypatch.setattr(runner, 'execute', blocked)
    args = SimpleNamespace(output_dir=root, wheel=tmp_path / 'x',
                           expected_version='1', phase_timeout=1)
    assert runner.run(args) == 1
    report = runner.read(root / 'summary.json')
    assert phases == (['negative', 'fit', 'score', 'evaluate']
                      if ready else ['negative'])
    assert report['status'] == 'blocked'
    assert report['phases']['negative']['status'] == ('passed'
                                                      if ready else 'blocked')
    assert report['phases']['fit']['status'] == ('passed'
                                                 if ready else 'blocked')
    assert report['required_gaps']['exported_epoch_history'][
        'status'] == 'not implemented'


@pytest.mark.parametrize('dtype', ['bool', 'int32', 'Int64'])
def test_training_row_digest_preserves_column_types(dtype):
    original = pd.DataFrame({'id': pd.Series([0, 1], dtype='int64')})
    restored = original.astype({'id': dtype})
    assert runner.frame_digest(original) != runner.frame_digest(restored)
    assert runner.frame_digest(original) == runner.frame_digest(original.copy())


@pytest.mark.parametrize('fault', ['nan', 'bundle', 'registry', 'epochs'])
def test_local_fit_tracking_rejects_incomplete_evidence(fault):
    run = SimpleNamespace(info=SimpleNamespace(status='FINISHED'),
                          data=SimpleNamespace(tags={'kumo.bundle_id': 'b'}))
    history = {'train/loss': [(0, 1.0), (1, 0.5)]}
    registered = SimpleNamespace(tags={'kumo_bundle_id': 'b'})
    runner.local_fit_record(run, history, registered, 'b')
    if fault == 'nan':
        history['train/loss'][1] = (1, float('nan'))
    elif fault == 'bundle':
        run.data.tags.clear()
    elif fault == 'registry':
        registered.tags.clear()
    else:
        history['train/loss'] = [(0, 1.0)]
    with pytest.raises(AssertionError):
        runner.local_fit_record(run, history, registered, 'b')


@pytest.mark.parametrize('denied', [False, True])
def test_platform_worker_preserves_original_error_and_receipt(
        tmp_path, monkeypatch, denied):
    import importlib
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    helpers = importlib.import_module('wheel_mlops_platform')
    profile = SimpleNamespace(
        config=dict(environment='venv', platform='databricks'), model='test',
        experiment='test', artifact_root='/test', placement=lambda: {})
    monkeypatch.setattr(runner, 'platform_backend', lambda _: profile)
    monkeypatch.setattr(runner, 'platform_helpers', lambda: helpers)
    monkeypatch.setattr(runner, 'check_install', lambda *args: {})
    monkeypatch.setitem(sys.modules, 'torch',
                        SimpleNamespace(set_num_threads=lambda _: None))
    engine = SimpleNamespace(configure=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, 'kumo', SimpleNamespace(engine=engine))
    monkeypatch.setitem(sys.modules, 'kumo.engine', engine)

    def failing_fit(_):
        error = RuntimeError('original provider failure')
        if denied:
            error.status_code = 403
        raise error

    monkeypatch.setattr(runner, 'fit', failing_fit)
    args = SimpleNamespace(output_dir=tmp_path, wheel=tmp_path / 'wheel',
                           expected_version='test', phase='fit')
    assert runner.phase(args) == 1
    report = runner.read(tmp_path / 'fit.json')
    assert report['status'] == ('blocked' if denied else 'failed')
    assert 'original provider failure' in report['error']
    assert 'AttributeError' not in report['error']


def test_final_scratch_failure_overrides_all_passed_phase_receipts(
        tmp_path, monkeypatch):
    root = tmp_path / 'attempt'
    monkeypatch.setattr(runner, 'platform_backend', lambda _: None)
    monkeypatch.setattr(runner, 'check_install', lambda *args: {})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: None)

    def completed(command, cwd, log, timeout):
        phase = command[command.index('--phase') + 1]
        runner.write(cwd / (phase + '.json'), dict(status='passed'))
        if phase == runner.PHASES[-1]:
            (cwd / ('scratch-' + phase)).mkdir()
        return 0

    def cannot_remove(path):
        raise OSError('Cannot remove final producer scratch')

    monkeypatch.setattr(runner, 'execute', completed)
    monkeypatch.setattr(runner.shutil, 'rmtree', cannot_remove)
    args = SimpleNamespace(output_dir=root, wheel=tmp_path / 'x',
                           expected_version='1', phase_timeout=1)
    assert runner.run(args) == 1
    report = runner.read(root / 'summary.json')
    assert report['status'] == 'failed'
    assert all(phase['status'] == 'passed'
               for phase in report['phases'].values())
    assert 'Cannot remove final producer scratch' in report['error']


def customer_module(tmp_path, phases="('train', 'recover')",
                    body='return dict(rows=3)'):
    path = tmp_path / 'customer.py'
    path.write_text(
        f'PHASES = {phases}\ndef run_phase(name, root, profile):\n    {body}\n')
    return path


@pytest.mark.parametrize('phases', [
    "()", "('train', 'train')", "('../escape',)", "('platform-prerequisite',)",
    "('summary',)", "('platform',)", "'train'", '(42,)'
])
def test_invalid_customer_phases_are_rejected(tmp_path, phases):
    args = SimpleNamespace(workload=customer_module(tmp_path, phases))
    with pytest.raises(AssertionError):
        runner.workload(args)


def test_customer_workload_owns_phases_and_does_not_use_kumo_helpers(
        tmp_path, monkeypatch):
    path = customer_module(tmp_path)
    root = tmp_path / 'attempt'
    monkeypatch.setattr(runner, 'check_install', lambda *a, **kw: {})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: None)
    phases = []

    def completed(command, cwd, log, timeout):
        name = command[command.index('--phase') + 1]
        phases.append(name)
        assert command[command.index('--workload') + 1] == str(path)
        scratch = cwd / ('scratch-' + name)
        scratch.mkdir()
        (scratch / 'temporary').touch()
        if name == 'recover':
            assert not (cwd / 'scratch-train').exists()
        runner.write(cwd / (name + '.json'), dict(status='passed'))
        return 0

    monkeypatch.setattr(runner, 'execute', completed)
    args = SimpleNamespace(output_dir=root, wheel=tmp_path / 'x',
                           expected_version='1', phase_timeout=1, workload=path)
    assert runner.run(args) == 0
    report = runner.read(root / 'summary.json')
    assert phases == ['train', 'recover']
    assert set(report['phases']) == set(phases)
    assert report['workload']['sha256'] == runner.digest(path)
    assert 'gpu_helpers_sha256' not in report
    assert not (root / 'scratch-recover').exists()


def test_customer_phase_does_not_configure_or_load_engine_trainer(
        tmp_path, monkeypatch):
    path = customer_module(tmp_path)
    monkeypatch.setattr(runner, 'check_install', lambda *a, **kw: {})
    # Custom code must not need Engine configuration, metrics, or model types.
    monkeypatch.setitem(sys.modules, 'kumo.engine', None)
    args = SimpleNamespace(output_dir=tmp_path, wheel=tmp_path / 'x',
                           expected_version='1', phase='recover', workload=path)
    assert runner.phase(args) == 0
    report = runner.read(tmp_path / 'recover.json')
    assert report['result'] == {'rows': 3}
    assert report['resources']['torch_threads'] is None


@pytest.mark.parametrize('ready', [True, False])
def test_custom_platform_workload_preserves_permission_gate(
        tmp_path, monkeypatch, ready):
    path = customer_module(tmp_path)
    profile = SimpleNamespace(
        config=dict(environment='managed', execution_context='sagemaker'))
    monkeypatch.setattr(runner, 'platform_backend', lambda _: profile)
    monkeypatch.setattr(runner, 'check_install', lambda *a, **kw: {})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: None)
    phases = []

    def completed(command, cwd, log, timeout):
        name = command[command.index('--phase') + 1]
        assert '--execute-platform' in command
        phases.append(name)
        runner.write(cwd / (name + '.json'),
                     dict(status='passed' if ready else 'blocked'))
        return 0 if ready else 1

    monkeypatch.setattr(runner, 'execute', completed)
    args = SimpleNamespace(output_dir=tmp_path / 'attempt',
                           wheel=tmp_path / 'x', expected_version='1',
                           phase_timeout=1, workload=path)
    assert runner.run(args) == 1  # UI review remains required.
    report = runner.read(args.output_dir / 'summary.json')
    assert phases == (['platform-prerequisite', 'train', 'recover']
                      if ready else ['platform-prerequisite'])
    assert 'exported_epoch_history' not in report['required_gaps']
    assert report['required_gaps']['ui_review']['status'] == 'blocked'


def test_custom_exception_produces_failed_receipt(tmp_path, monkeypatch):
    path = customer_module(tmp_path,
                           body="raise ValueError('bad customer weights')")
    monkeypatch.setattr(runner, 'check_install', lambda *a, **kw: {})
    args = SimpleNamespace(output_dir=tmp_path, wheel=tmp_path / 'x',
                           expected_version='1', phase='recover', workload=path)
    assert runner.phase(args) == 1
    receipt = runner.read(tmp_path / 'recover.json')
    assert receipt['status'] == 'failed'
    assert 'bad customer weights' in receipt['error']


def test_custom_phase_runs_in_a_fresh_isolated_process(tmp_path):
    path = customer_module(
        tmp_path, body="return dict(previous=(root / 'train.json').is_file())")
    # Exercise actual -I imports and receipts without installing a candidate wheel.
    program = (
        'import importlib.util; from pathlib import Path; from types import SimpleNamespace; '
        f's=importlib.util.spec_from_file_location("runner", {str(SCRIPT)!r}); '
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
        'm.check_install=lambda *a, **kw: {}; '
        f'a=SimpleNamespace(workload=Path({str(path)!r}), output_dir=Path({str(tmp_path)!r}), '
        'wheel=Path("unused"), expected_version="1", phase=PHASE); '
        'raise SystemExit(m.phase(a))')
    for name in ('train', 'recover'):
        subprocess.run(
            [sys.executable, '-I', '-c',
             program.replace('PHASE', repr(name))], check=True)
    assert runner.read(tmp_path / 'recover.json')['result']['previous'] is True
