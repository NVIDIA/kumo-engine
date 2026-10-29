#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the exact candidate artifact and run selected PR checks in isolation."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import runpy
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from contextlib import chdir, contextmanager, nullcontext
from email.parser import Parser
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    'binary', 'multiclass', 'multilabel', 'multilabel_ranking', 'regression',
    'forecasting', 'static_link', 'temporal_link'
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_identity(repo, revision):
    def git(*args):
        return subprocess.check_output(
            ['git', '-C', str(repo), *args], text=True).strip()

    if (not re.fullmatch('[0-9a-f]{40}', revision)
            or git('rev-parse', 'HEAD') != revision
            or git('status', '--porcelain', '--untracked-files=normal')):
        raise ValueError(
            'Build requires a clean checkout of the exact revision')
    version = (repo / 'VERSION').read_text().strip()
    return dict(
        revision=revision, tree=git('rev-parse', 'HEAD^{tree}'),
        bundle_version=f"{version}+{git('rev-parse', '--short', 'HEAD')}")


def wheel_metadata(wheel):
    with ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith('.dist-info/METADATA')
        ]
        if len(names) != 1:
            raise ValueError('Expected exactly one wheel distribution')
        fields = Parser().parsestr(archive.read(names[0]).decode())
        if fields['Name'] != 'kumo-engine' or not fields['Version']:
            raise ValueError('Expected a versioned kumo-engine wheel')
        return fields['Version']


def build_receipt(source, wheel):
    version = wheel_metadata(wheel)
    if version != source['bundle_version']:
        raise ValueError('The wheel version does not identify this PR revision')
    return dict(source, version=version, wheel_sha256=digest(wheel),
                wheel=wheel.name)


def verify_installed_files(wheel, location):
    count = 0
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/') or '.dist-info/' in name:
                continue
            destination = (location / name).resolve()
            if (not destination.is_relative_to(location.resolve())
                    or not destination.is_file() or digest(destination) !=
                    hashlib.sha256(archive.read(name)).hexdigest()):
                raise ValueError(f'Installed candidate member differs: {name}')
            count += 1
    if not count:
        raise ValueError('Candidate archive contains no runtime files')
    return count


def verify_roster(report):
    rows = report.get('runs', [])
    if (report.get('status') != 'MEASURED'
            or report.get('learning_quality_policy') != 'report_only'
            or len(rows) != 8 or {(r['task'], r['seed'], r['case'])
                                  for r in rows} != {(t, 0, 'learnable')
                                                     for t in TASKS}):
        raise ValueError(
            'PR requires eight normal seed-zero fits and score reporting')


def verify_fits(rows):
    if not rows or any(
            r.get('status') != 'PASS' or r.get('device') != 'cuda:0'
            or type(r.get('steps')) is not int or r['steps'] <= 0
            or type(r.get('cuda_batches')) is not int or r['cuda_batches'] <= 0
            for r in rows):
        raise ValueError('Every fit requires completed CUDA optimizer updates')


def verify_b7(output, version):
    paths = list(
        output.glob('pytest/test_retained_tables_and_model*/child/result.json'))
    if len(paths) != 1 or paths[0].is_symlink():
        raise ValueError('Require one retained-model B7 child receipt')
    path = paths[0]
    child = json.loads(path.read_text())
    counts = ('prediction_batches', 'cuda_prediction_batches', 'entities',
              'catalog', 'requested', 'returned_per_entity', 'rows',
              'new_entity', 'new_item')
    states = child.get('state_checks', [])
    if (child.get('status') != 'PASS' or child.get('version') != version
            or child.get('requested_accelerator') != 'gpu'
            or type(child.get('fits_started')) is not int
            or child['fits_started'] != 0 or any(
                type(child.get(name)) is not int or child[name] <= 0
                for name in counts)
            or child['cuda_prediction_batches'] != child['prediction_batches']
            or child.get('observed_devices') !=
        ['cuda:0'] * child['prediction_batches']
            or child['requested'] != child['catalog']
            or child['returned_per_entity'] != child['catalog']
            or child['rows'] != child['entities'] * child['catalog']
            or child['entities'] != child['new_entity'] + 1
            or child['catalog'] != child['new_item'] + 1 or any(
                child.get(name) is not True
                for name in ('source_unchanged', 'bundle_unchanged')) or
            any(not isinstance(child.get(name), str) or not child[name]
                for name in ('snapshot_id', 'training_snapshot_id', 'anchor'))
            or child['snapshot_id'] == child['training_snapshot_id'] or
            not re.fullmatch('[0-9a-f]{64}', child.get('checkpoint_sha256', ''))
            or [state.get('phase')
                for state in states] != ['post_restore', 'post_prediction']):
        raise ValueError('B7 lacks complete unchanged-state CUDA model reuse')
    before, after = [{
        key: value
        for key, value in state.items() if key != 'phase'
    } for state in states]
    remapped = before.get('remapped_state_entries', [])
    if (before != after or before.get('device') != 'cuda:0'
            or before.get('lookup_mode') != 'lookup'
            or any(not isinstance(before.get(name), str) or not before[name]
                   for name in ('module', 'lookup'))
            or any(
                type(before.get(name)) is not int
                for name in ('old_entities_preserved', 'new_entities',
                             'reindexed_entities', 'checkpoint_state_entries'))
            or before['old_entities_preserved'] != child['catalog'] - 1
            or before['new_entities'] != 1 or not 0 <=
            before['reindexed_entities'] <= before['old_entities_preserved']
            or before['checkpoint_state_entries'] <= 3
            or len(set(remapped)) != 3 or not remapped or remapped != [
                remapped[0].removesuffix('weight') + suffix
                for suffix in ('weight', 'mask', '_extra_state')
            ] or 'rhs_embedding.' not in remapped[0]
            or not remapped[0].endswith('lookup_embedding.weight')):
        raise ValueError(
            'B7 restored and post-prediction state evidence differs')
    xml = output / 'tests.xml'
    tree = ET.parse(xml)
    cases = list(tree.iter('testcase'))
    required = [
        ('test_pr_behavior',
         'test_retained_tables_and_model_predict_in_fresh_process'),
        ('test_training_table_export', 'test_node_export_roundtrip[temporal-'),
        ('test_training_table_export', 'test_node_export_roundtrip[static-')
    ]
    if (any(
            case.find(tag) is not None for case in cases
            for tag in ('failure', 'error', 'skipped')) or any(
                int(suite.get(name, 0)) for suite in tree.iter('testsuite')
                for name in ('failures', 'errors', 'skipped')) or any(
                    sum(
                        case.get('classname', '').split('.')[-1] == module
                        and case.get('name', '').startswith(name)
                        for case in cases) != 1 for module, name in required)):
        raise ValueError(
            'B7 requires passing retained-model and weighted export tests')
    return dict(status='PASS', version=version,
                child=path.relative_to(output).as_posix(),
                child_sha256=digest(path), junit_sha256=digest(xml))


def correctness_coverage(root, version):
    phases = {}
    for name in ('install', 'example', 'behavior'):
        report = json.loads((root / name / 'result.json').read_text())
        fits = json.loads((root / name / 'fits.json').read_text())
        verify_fits(fits)
        if (report.get('status') != 'PASS'
                or type(report.get('physical_fits')) is not int
                or report['physical_fits'] != len(fits)):
            raise ValueError(f'Required {name} evidence failed or differs')
        if name == 'behavior' and report.get('b7') != verify_b7(
                root / name, version):
            raise ValueError('The required B7 marker is missing or differs')
        phases[name] = dict(status='PASS', physical_fits=len(fits))
    return dict(
        status='PASS', phases=phases, b7='PASS',
        additional_physical_fits=sum(phase['physical_fits']
                                     for phase in phases.values()))


def verify_quickstart_predictions(frame, ids):
    import numpy as np

    probabilities = frame[['False_PROB', 'True_PROB']].to_numpy()
    fixture = json.loads((
        ROOT /
        'packages/kumo-engine/test/wheel_workflow.json').read_text())['fixture']
    # Reuse the existing binary output contract, never a prediction-delta gate.
    normalized = np.allclose(probabilities.sum(axis=1), 1,
                             atol=fixture['reload_atol'],
                             rtol=fixture['reload_rtol'])
    if (not frame['ENTITY'].is_unique or set(frame['ENTITY']) != set(ids)
            or not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
            or not normalized
            or not np.array_equal(frame['TARGET_PRED'],
                                  probabilities[:, 1] > fixture['threshold'])):
        raise ValueError('Quickstart predictions are incomplete or invalid')


def prediction_tensors(outputs):
    values = (outputs.get('logits', outputs.get('score')) if isinstance(
        outputs, dict) else outputs)
    return list(values.values()) if isinstance(values, dict) else [values]


@contextmanager
def quickstart_prediction_check(output):
    import pyarrow.parquet as pq

    import kumo.engine as kumo
    from kumo.engine import datasets

    ids = pq.read_table(Path(datasets.shop()) /
                        'users.parquet')['user_id'].to_pylist()
    original = kumo.Trainer.predict

    def predict(trainer, *args, **kwargs):
        result = original(trainer, *args, **kwargs)
        frame = result.frame
        verify_quickstart_predictions(frame, ids)
        frame.to_parquet(output / 'predictions.parquet', index=False)
        return result

    with patch.object(kumo.Trainer, 'predict', predict):
        yield


@contextmanager
def count_fits(path):
    import torch
    from kumo_core.train.trainer import Trainer
    from pytorch_lightning import Callback

    original = Trainer.fit
    original_create = Trainer.create_pl_trainer
    rows = []
    batches = dict(training=0, prediction=0)

    class CompletedCudaWork(Callback):
        def check(self, trainer, tensor):
            if (str(trainer.strategy.root_device) != 'cuda:0'
                    or not isinstance(tensor, torch.Tensor)
                    or not tensor.numel() or str(tensor.device) != 'cuda:0'):
                raise ValueError('Missing completed CUDA tensor work')
            torch.cuda.synchronize()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            self.check(
                trainer,
                outputs.get('loss') if isinstance(outputs, dict) else outputs)
            batches['training'] += 1
            rows[-1]['cuda_batches'] += 1
            save(path, rows)

        def on_predict_batch_end(self, trainer, model, outputs, batch,
                                 batch_idx, dataloader_idx=0):
            tensors = prediction_tensors(outputs)
            if not tensors:
                raise ValueError('Prediction returned no CUDA tensor work')
            for tensor in tensors:
                self.check(trainer, tensor)
            batches['prediction'] += 1
            save(path.with_name('cuda-batches.json'), batches)

    def create(trainer):
        result = original_create(trainer)
        trainer.pl_trainer.callbacks.append(CompletedCudaWork())
        return result

    def fit(trainer, *args, **kwargs):
        row = dict(status='RUNNING', cuda_batches=0)
        rows.append(row)
        save(path, rows)
        started = time.monotonic()
        try:
            result = original(trainer, *args, **kwargs)
            row.update(status='PASS',
                       device=str(trainer.pl_trainer.strategy.root_device),
                       steps=int(trainer.pl_trainer.global_step))
            verify_fits([row])
            return result
        except BaseException:
            row['status'] = 'FAIL'
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            save(path, rows)

    Trainer.fit = fit
    Trainer.create_pl_trainer = create
    try:
        yield rows
        if not batches['training'] or not batches['prediction']:
            raise ValueError(
                'Selected checks need actual training and prediction CUDA batches'
            )
    finally:
        Trainer.fit = original
        Trainer.create_pl_trainer = original_create
        save(path.with_name('cuda-batches.json'), batches)


def run_logged(command, output, *, cwd):
    started = time.monotonic()
    with output.open('w') as stream:
        completed = subprocess.run(command, cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'Command failed ({completed.returncode}): {output}')
    return dict(command=command, seconds=time.monotonic() - started)


def run_phase(args):
    import pytest
    import torch
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'work'
    work.mkdir()
    modules = ROOT / 'packages/kumo-engine/test'
    sys.path.insert(0, str(modules))
    from wheel_gpu_learning import (
        observe_gpu,
        verify_devices,
        verify_reload_identity,
    )
    torch.set_num_threads(4)
    report = dict(status='RUNNING', phase=args.phase)
    try:
        with count_fits(output / 'fits.json') as fits, chdir(work), \
                observe_gpu(report, strict_batches=True):
            if args.phase == 'install':
                sys.argv = [
                    str(modules / 'wheel_workflow.py'), '--version',
                    args.version, '--device', 'cuda', '--measure-variation',
                    '--output-dir',
                    str(output / 'workflow')
                ]
                try:
                    runpy.run_path(sys.argv[0], run_name='__main__')
                except SystemExit as error:
                    if error.code not in (0, None):
                        raise
                result = json.loads(
                    (output / 'workflow/workflow-result.json').read_text())
                if result['status'] != 'MEASURED':
                    raise ValueError('Installation workflow did not complete')
                verify_devices(result,
                               ('training', 'prediction', 'reload_and_predict'))
                verify_reload_identity(result)
                if result['checks'].get(
                        'saved_and_loaded_model_identity') != 'PASS':
                    raise ValueError(
                        'Installation workflow lacks loaded-state identity')
            else:
                paths = ([
                    'test_readme_quickstart.py'
                ] if args.phase == 'example' else [
                    'test_pr_behavior.py', 'test_warm_start.py',
                    'test_training_table_export.py',
                    'test_static_link_split.py',
                    'test_static_link_pseudo_time.py::test_static_link_subgraphs_hide_later_split_labels',
                    'test_static_link_pseudo_time.py::test_native_iterator_infers_cutoff_without_loader_options',
                    'test_static_link_pseudo_time.py::test_stage_visibility_overrides_generic_runtime_options',
                    'test_static_query_time_cutoff.py',
                    'test_timestamp_batch_cap.py',
                    'test_prediction_workers.py::test_bad_worker_count_fails_at_public_entry',
                    'test_prediction_workers.py::test_bad_class_count_fails_before_prediction_work'
                ])
                # Package-local fixtures import the installed wheel. No root
                # conftest or source package path is added to this interpreter.
                with (quickstart_prediction_check(output)
                      if args.phase == 'example' else nullcontext()):
                    code = pytest.main([
                        '-q', '-c', '/dev/null', '--import-mode=importlib',
                        '--basetemp', str(output / 'pytest'),
                        '--confcutdir',
                        str(modules), '--junitxml',
                        str(output / 'tests.xml'),
                        *[str(modules / p) for p in paths]
                    ])
                if code:
                    raise ValueError(f'{args.phase} tests failed ({code})')
                xml = ET.parse(output / 'tests.xml')
                suites = list(xml.iter('testsuite'))
                if (not suites
                        or sum(int(s.get('tests', 0)) for s in suites) == 0
                        or any(int(s.get('skipped', 0)) for s in suites)):
                    raise ValueError(
                        'Selected behavior checks cannot be empty or skipped')
            verify_fits(fits)
            if args.phase == 'behavior':
                report['b7'] = verify_b7(output, args.version)
            report.update(status='PASS', physical_fits=len(fits))
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        if (output / 'fits.json').exists():
            report['physical_fits'] = len(
                json.loads((output / 'fits.json').read_text()))
        save(output / 'result.json', report)


def verify_install(build, wheel):
    import torch

    import kumo.engine as kumo
    if (build['wheel_sha256'] != digest(wheel)
            or wheel_metadata(wheel) != build['version']
            or importlib.metadata.version('kumo-engine') != build['version']
            or kumo.__version__ != build['version'] or
            not Path(kumo.__file__).resolve().is_relative_to(Path(sys.prefix))
            or sys.platform != 'linux' or sys.version_info[:2] != (3, 12)
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError(
            'Use the exact installed Linux Python3.12 wheel and one CUDA GPU')
    distribution = importlib.metadata.distribution('kumo-engine')
    files = verify_installed_files(wheel, Path(distribution.locate_file('')))
    return dict(version=build['version'], wheel_sha256=build['wheel_sha256'],
                installed_members_verified=files, import_path=kumo.__file__,
                python=sys.version, gpu=torch.cuda.get_device_name(0),
                cuda=torch.version.cuda)


def run_checks(args):
    selection = json.loads(args.selection.read_text())
    build = json.loads(args.build.read_text())
    if selection['revision'] != build['revision']:
        raise ValueError('Selection and built revision differ')
    actual = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != build['revision']:
        raise ValueError('Worker checkout differs from built revision')
    [wheel] = list(args.wheel_dir.glob('*.whl'))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'work'
    work.mkdir()
    report = dict(status='RUNNING', revision=build['revision'],
                  learning_scores_block=False, stages=[], primary_fits=0,
                  additional_fits=0, full_ci_time_verified=False)
    started = time.monotonic()
    modules = ROOT / 'packages/kumo-engine/test'
    python = sys.executable
    try:
        report['installation'] = verify_install(build, wheel)
        run_logged([python, '-I', '-m', 'pip', '--isolated', 'check'],
                   output / 'pip-check.log', cwd=work)
        packages = {
            re.sub(r'[-_.]+', '-', d.metadata['Name']).lower(): d.version
            for d in importlib.metadata.distributions() if
            re.sub(r'[-_.]+', '-', d.metadata['Name']).lower() != 'kumo-engine'
        }
        dependencies = output / 'dependencies.json'
        save(dependencies, packages)
        if selection['learning']:
            command = [
                python, '-I', '-u',
                str(modules / 'wheel_gpu_learning.py'), '--version',
                build['version'], '--device', 'cuda', '--group', 'all',
                '--seed-profile', 'pr', '--scale-bench-rung', 'pr',
                '--scale-bench-data',
                str(args.data.resolve()), '--scale-bench-source',
                str(args.scale_source.resolve()), '--dependency-profile',
                str(dependencies), '--output-dir',
                str(output / 'learning')
            ]
            argv = output / 'primary-arguments.json'
            save(argv, command[4:])
            command = [
                python, '-I', '-u',
                str(Path(__file__).resolve()), 'primary', '--arguments',
                str(argv), '--output',
                str(output / 'primary-fits.json')
            ]
            report['stages'].append(
                run_logged(command, output / 'learning.log', cwd=work))
            measured = json.loads(
                (output / 'learning/scale-result.json').read_text())
            verify_roster(measured)
            report['primary_fits'] = len(measured['runs'])
            report['scores'] = [{
                key: row[key]
                for key in ('task', 'metric', 'score')
            } for row in measured['runs']]
        for phase in ('install', 'example', 'behavior'):
            if not selection[phase]:
                continue
            command = [
                python, '-I', '-u',
                str(Path(__file__).resolve()), 'phase', '--phase', phase,
                '--version', build['version'], '--output',
                str(output / phase)
            ]
            report['stages'].append(
                run_logged(command, output / f'{phase}.log', cwd=work))
            result = json.loads((output / phase / 'result.json').read_text())
            if result['status'] != 'PASS':
                raise ValueError(f'{phase} did not complete')
            report['additional_fits'] += result['physical_fits']
        report['final_installation'] = verify_install(build, wheel)
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        ledger = output / 'primary-fits.json'
        report['primary_fit_attempts'] = (len(json.loads(ledger.read_text()))
                                          if ledger.exists() else 0)
        report['additional_fit_attempts'] = sum(
            len(json.loads(p.read_text())) for p in output.glob('*/fits.json'))
        report['physical_fits'] = report['primary_fits'] + report[
            'additional_fits']
        save(output / 'result.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    source = commands.add_parser('source')
    source.add_argument('--revision', required=True)
    source.add_argument('--output', type=Path, required=True)
    built = commands.add_parser('record-build')
    built.add_argument('--source', type=Path, required=True)
    built.add_argument('--wheel-dir', type=Path, required=True)
    built.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--selection', type=Path, required=True)
    run.add_argument('--build', type=Path, required=True)
    run.add_argument('--wheel-dir', type=Path, required=True)
    run.add_argument('--data', type=Path)
    run.add_argument('--scale-source', type=Path)
    run.add_argument('--output', type=Path, required=True)
    phase = commands.add_parser('phase')
    phase.add_argument('--phase', choices=('install', 'example', 'behavior'),
                       required=True)
    phase.add_argument('--version', required=True)
    phase.add_argument('--output', type=Path, required=True)
    primary = commands.add_parser('primary')
    primary.add_argument('--arguments', type=Path, required=True)
    primary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'source':
        save(args.output, source_identity(ROOT, args.revision))
    elif args.command == 'record-build':
        source = json.loads(args.source.read_text())
        if source_identity(ROOT, source['revision']) != source:
            raise ValueError('Source tree changed during the build')
        [wheel] = list(args.wheel_dir.glob('*.whl'))
        save(args.output, build_receipt(source, wheel))
    elif args.command == 'phase':
        run_phase(args)
    elif args.command == 'primary':
        worker = ROOT / 'packages/kumo-engine/test/wheel_gpu_learning.py'
        arguments = json.loads(args.arguments.read_text())
        with count_fits(args.output) as fits:
            sys.argv = [str(worker), *arguments]
            try:
                runpy.run_path(str(worker), run_name='__main__')
            except SystemExit as error:
                if error.code not in (0, None):
                    raise
            verify_fits(fits)
            if len(fits) != 8:
                raise ValueError(
                    'Primary worker must perform exactly eight fits')
    else:
        run_checks(args)


if __name__ == '__main__':
    main()
