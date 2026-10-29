#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing fixed learning recipes on one observed CUDA device."""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import runpy
import sys
import time
import traceback
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def verify_dependencies(packages, profile):
    if packages != json.loads(profile.read_text()):
        raise ValueError(
            'Installed dependencies differ from the frozen profile')


def source_identity(*, prepared=False):
    tasks = runpy.run_path(str(ROOT / 'wheel_task_learning.py'))
    names = (*tasks['SOURCES'], 'wheel_gpu_learning.py')
    if prepared:
        names += ('wheel_scale_data.py', 'wheel_scale_learning.py',
                  'wheel_measurements.py')
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in names
    }
    return hashlib.sha256(json.dumps(sources,
                                     sort_keys=True).encode()).hexdigest()


def verify_devices(result, required_stages):
    observed = result.get('observed_devices', [])
    if (result.get('device') != 'cuda' or not observed or any(
            row.get('device') not in ('cuda', 'cuda:0') for row in observed)
            or not set(required_stages).issubset(
                row.get('stage') for row in observed)):
        raise ValueError('Missing CUDA execution evidence or CPU fallback')


def verify_report(report, *, additional, prepared=False):
    if (report.get('configuration', {}).get('device') != 'cuda'
            or report.get('gpu_source_sha256')
            != source_identity(prepared=prepared)):
        raise ValueError('GPU runtime or guard source differs')
    for row in report.get('runs', []):
        verify_devices(row, ('training', 'prediction',
                             'reload_and_predict') if additional else
                       ('training', ))
        if additional:
            measured = row.get('consistency_measurements', [])
            stages = (['prediction', 'prediction', 'reload_and_predict']
                      if prepared else ['prediction', 'reload_and_predict'])
            if (len(measured) != len(stages)
                    or [m.get('stage') for m in measured] != stages
                    or prepared and [m.get('scope') for m in measured
                                     ] != ['repeat_full', 'partial', 'reload']):
                raise ValueError(
                    'Required GPU consistency measurements are missing or misidentified'
                )
            for item in measured:
                value = item.get('max_abs_difference')
                finite = (type(value) in (int, float) and math.isfinite(value)
                          and value >= 0)
                if row.get('target_format') == 'ranked_link_v1':
                    population = (max(1,
                                      len(row['entity_ids']) //
                                      2) if item['scope'] == 'partial' else len(
                                          row['entity_ids']))
                    total = sum(map(len,
                                    row.get('ranked_ids', [])[:population]))
                    counts = [
                        item.get(name) for name in ('common_recommendations',
                                                    'removed_recommendations',
                                                    'added_recommendations',
                                                    'changed_entities')
                    ]
                    if (item.get('score_coordinates') != 'entity_class' or any(
                            type(count) is not int or count < 0
                            for count in counts)
                            or counts[0] + counts[1] != total
                            or counts[0] + counts[2] != total
                            or counts[3] > population):
                        raise ValueError(
                            'Invalid ranked consistency coordinates')
                    finite = (finite and counts[0] > 0
                              or value is None and counts[0] == 0)
                if (item.get('status') != 'MEASURED'
                        or item.get('qualified') is not False
                        or item.get('gpu_tolerance') is not None or not finite):
                    raise ValueError('Invalid GPU consistency measurement')
            if prepared:
                repeat = row.get('repeat_max_abs_difference')
                valid = (type(repeat) in (int, float)
                         or row.get('target_format') == 'ranked_link_v1'
                         and repeat is None)
                if not valid or repeat != measured[0]['max_abs_difference']:
                    raise ValueError('Repeated full prediction delta differs '
                                     'from its measurement')
    if not report.get('runs'):
        raise ValueError('GPU measurements contain no completed models')


SEED_PROFILES = {'pr': (0, ), 'full': (0, 1, 2, 3, 4)}


def verify_prepared_report(report, *, calibration_dir=None, expected_runs=None):
    import wheel_scale_learning as learning
    import wheel_task_checks as checks
    from wheel_measurements import verify_training_counts

    cases = runpy.run_path(str(ROOT / 'wheel_scale_data.py'))['CASES']
    group = report.get('group')
    if group not in ('binary', 'additional', 'all'):
        raise ValueError('Unknown prepared-profile group')
    tasks = [
        name for name in cases
        if group == 'all' or (name == 'binary') == (group == 'binary')
    ]
    profile = report.get('configuration', {}).get('seed_profile')
    if expected_runs is None:
        seeds = SEED_PROFILES.get(profile)
        if seeds is None:
            raise ValueError('A fixed PR or full seed profile is required')
        expected = {
            (task, seed, case)
            for task in tasks
            for seed in seeds
            for case in (('learnable', ) if seeds == (0, ) else ('learnable',
                                                                 'independent'))
        }
    else:
        roster = list(expected_runs)
        if (profile != 'nightly' or calibration_dir is not None or not roster
                or any(not isinstance(item, (tuple, list)) or len(item) != 3
                       or type(item[0]) is not str or item[0] not in tasks
                       or type(item[1]) is not int or item[1] != 0
                       or item[2] not in ('learnable', 'independent')
                       for item in roster)):
            raise ValueError('Nightly needs an explicit fixed seed-zero roster')
        expected = {tuple(item) for item in roster}
        if len(expected) != len(roster):
            raise ValueError('Nightly cannot repeat requested cases')
        seeds = (0, )
    rows = report.get('runs', [])
    fault = report.get('calibration_fault')
    if calibration_dir is not None:
        if (not isinstance(fault, dict) or set(fault) != {'kind', 'budget'}
                or fault['kind'] != 'reduced_training_budget'
                or report.get('release_approved') is not False):
            raise ValueError('Calibration cannot approve release')
    elif ('calibration_fault' in report
          or any('calibration_fault' in row for row in rows)):
        raise ValueError('A calibration run cannot approve release')
    if (report.get('status') != 'MEASURED' or report.get('seeds') != list(seeds)
            or len(rows) != len(expected)
            or {(row['task'], row['seed'], row['case'])
                for row in rows} != expected
            or len({row.get('fit_id')
                    for row in rows}) != len(rows) or
            any(not row.get('fit_id') or row.get('status') != 'MEASURED' or row.
                get('batch_retries_disabled') is not True or type(row['seed'])
                is not int or type(row.get('training_seed')) is not int
                or row.get('training_seed') != row['seed'] or any(
                    row.get('checks', {}).get(check) != 'PASS'
                    for check in ('fit', 'prediction', 'repeat_prediction',
                                  'partial_prediction', 'reload',
                                  'hidden_answers', 'chronology'))
                for row in rows)):
        raise ValueError('Missing, repeated, or failed prepared-profile models')
    verify_report(report, additional=True, prepared=True)
    for row in rows:
        ranked = row.get('target_format') == 'ranked_link_v1'
        expected_ranked = (report.get('configuration', {}).get('ranked_links')
                           is True and row['task'].endswith('link'))
        if ranked != expected_ranked:
            raise ValueError('Prepared prediction representation differs')
        verify_reload_identity(row)
        verify_training_counts(
            row,
            budget=fault['budget'] if calibration_dir is not None else None)
        if calibration_dir is not None:
            learning.verify_calibration_plan(
                row,
                Path(calibration_dir) /
                f'{row["task"]}-{row["seed"]}-{row["case"]}', fault)
        if (type(row.get('selected_checkpoint_global_step')) is not int
                or row['selected_checkpoint_global_step'] <= 0
                or row['selected_checkpoint_global_step']
                > row['training_work'][0]['optimizer_steps']):
            raise ValueError('Prepared model has no positive training step')
        spec = cases[row['task']]
        stages = row.get('stage_sha256', {})
        digests = [
            row.get('data_sha256'),
            row.get('plan_sha256'), *stages.values()
        ]
        if ((row.get('metric'), row.get('task_type'))
                != (spec['metric'], spec['task_type'])
                or set(stages) != {'train', 'val', 'test'}
                or any(not isinstance(value, str)
                       or not re.fullmatch('[0-9a-f]{64}', value)
                       for value in digests)):
            raise ValueError('Prepared task metadata or fingerprints differ')
        for field, values in (('score', 'predictions'),
                              ('baseline_score', 'baseline_predictions')):
            if ranked:
                selected = (dict(row, ranked_ids=row['baseline_ranked_ids'],
                                 ranked_scores=row['baseline_ranked_scores'])
                            if field == 'baseline_score' else row)
                measured = checks.ranked_link_score(selected)
            else:
                checks.verify_predictions(row['entity_ids'], row['entity_ids'],
                                          row[values])
                measured = learning.task_score(row['task'], row['targets'],
                                               row[values])
            if (type(row.get(field)) not in (int, float)
                    or not math.isfinite(row[field]) or not math.isclose(
                        measured, row[field], abs_tol=1e-12, rel_tol=0)):
                raise ValueError(
                    'Prepared score differs from saved predictions')


def measure_consistency(expected, actual):
    import numpy as np

    expected, actual = np.asarray(expected,
                                  dtype=float), np.asarray(actual, dtype=float)
    if (expected.ndim != 2 or not expected.size
            or expected.shape != actual.shape
            or not np.isfinite(expected).all()
            or not np.isfinite(actual).all()):
        raise ValueError('Prediction matrices must have equal finite shapes')
    return dict(
        status='MEASURED', qualified=False, gpu_tolerance=None,
        max_abs_difference=float(np.max(np.abs(expected - actual))),
        cpu_tolerance_exceeded=not bool(
            np.allclose(expected, actual, atol=1e-6, rtol=1e-6)))


def checkpoint_sha256(path):
    from kumo_core.io.filesystem import local_path

    with local_path(str(path)).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def verify_checkpoint_extra_state(expected, actual):
    import pandas as pd

    if type(expected) is not type(actual):
        raise ValueError('Reload extra state types differ')
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError('Reload extra state keys differ')
        for key in expected:
            verify_checkpoint_extra_state(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise ValueError('Reload extra state lengths differ')
        for left, right in zip(expected, actual):
            verify_checkpoint_extra_state(left, right)
    elif isinstance(expected, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        except AssertionError as error:
            raise ValueError('Reload extra state frame differs') from error
    elif expected is None or type(expected) in (bool, int, float, str):
        if expected != actual:
            raise ValueError('Reload extra state value differs')
    else:
        raise ValueError(f'Unsupported reload extra state: {type(expected)}')


def verify_loaded_checkpoint(model, checkpoint_path, expected_sha256):
    import torch
    from kumo_core.train.checkpoint import load_kumo_checkpoint

    if checkpoint_path is None:
        raise ValueError('Reload checkpoint path is missing')
    if checkpoint_sha256(checkpoint_path) != expected_sha256:
        raise ValueError('Reload checkpoint fingerprint differs')
    expected = load_kumo_checkpoint(checkpoint_path,
                                    map_location='cpu')['state_dict']
    actual = model.state_dict()
    if set(actual) != set(expected):
        raise ValueError('Reload state keys differ')
    tensors = 0
    for key in expected:
        left, right = expected[key], actual[key]
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if (not isinstance(left, torch.Tensor)
                    or not isinstance(right, torch.Tensor)
                    or left.shape != right.shape or left.dtype != right.dtype
                    or not torch.equal(left.detach().cpu(),
                                       right.detach().cpu())):
                raise ValueError(f'Reload tensor differs: {key}')
            tensors += 1
        else:
            verify_checkpoint_extra_state(left, right)
    if not tensors:
        raise ValueError('Reload checkpoint has no tensors')
    return dict(status='PASS', checkpoint_sha256=expected_sha256,
                checkpoint_path=str(checkpoint_path), tensor_count=tensors)


def verify_reload_identity(row):
    expected = row.get('expected_reload_checkpoint_sha256')
    receipts = row.get('reload_identity_checks')
    if (not isinstance(expected, str)
            or not re.fullmatch('[0-9a-f]{64}', expected)
            or not isinstance(receipts, list) or not receipts
            or any(not isinstance(receipt, dict) or receipt.get('status') !=
                   'PASS' or receipt.get('checkpoint_sha256') != expected
                   or receipt.get('restore_path') !=
                   'lightning_strategy.load_model_state_dict'
                   or not isinstance(receipt.get('checkpoint_path'), str)
                   or not receipt['checkpoint_path']
                   or type(receipt.get('tensor_count')) is not int
                   or receipt['tensor_count'] <= 0 for receipt in receipts)):
        raise ValueError('Missing or mismatched reload identity checks')


@contextmanager
def without_batch_retries(core, auto):
    def reject(*args, **kwargs):
        raise ValueError('A batch retry invalidates the fixed workload')

    with (patch.object(core, 'test', inspect.unwrap(core.test)),
          patch.object(core, 'predict_v2', inspect.unwrap(core.predict_v2)),
          patch.object(auto, '_handle_oom_retry', reject)):
        yield


@contextmanager
def observe_gpu(evidence, *, strict_batches=False):
    import torch
    from kumo_core.train.trainer import Trainer as CoreTrainer

    import kumo.engine as kumo

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one available CUDA GPU')
    evidence.update(device='cuda', observed_devices=[],
                    consistency_measurements=[])
    stage = 'initialization'
    loaded = set()
    original_create = CoreTrainer.create_pl_trainer
    original_init = kumo.Trainer.__init__
    original_fit = kumo.Trainer.fit
    original_predict = kumo.Trainer.predict
    original_load = kumo.Trainer.load
    original_run_path = runpy.run_path
    task_checks = importlib.import_module('wheel_task_checks')
    retries = nullcontext()
    if strict_batches:
        from kumo_core.automl.auto_trainer import AutoTrainer
        retries = without_batch_retries(CoreTrainer, AutoTrainer)

    def measured_reload(expected, actual):
        measured = measure_consistency(expected, actual)
        evidence['consistency_measurements'].append(
            dict(stage=stage, **measured))
        return measured['max_abs_difference']

    def load_checks(path, *args, **kwargs):
        result = original_run_path(path, *args, **kwargs)
        if Path(path).resolve() == ROOT / 'wheel_task_checks.py':
            result['verify_reload'] = measured_reload
        return result

    def create(trainer):
        result = original_create(trainer)
        device = str(trainer.pl_trainer.strategy.root_device)
        evidence['observed_devices'].append(dict(stage=stage, device=device))
        if device != 'cuda:0':
            raise ValueError(f'Requested cuda:0, observed {device}')
        if (stage == 'reload_and_predict'
                and 'expected_reload_checkpoint_sha256' in evidence):
            strategy = trainer.pl_trainer.strategy
            restore = strategy.load_model_state_dict

            def restore_model(*args, **kwargs):
                result = restore(*args, **kwargs)
                receipt = verify_loaded_checkpoint(
                    strategy.lightning_module, trainer.pl_trainer.ckpt_path,
                    evidence['expected_reload_checkpoint_sha256'])
                receipt[
                    'restore_path'] = 'lightning_strategy.load_model_state_dict'
                evidence.setdefault('reload_identity_checks',
                                    []).append(receipt)
                return result

            stack.enter_context(
                patch.object(strategy, 'load_model_state_dict', restore_model))
        return result

    def initialize(trainer, *args, **kwargs):
        if 'accelerator' in inspect.signature(original_init).parameters:
            kwargs['accelerator'] = 'gpu'
        original_init(trainer, *args, **kwargs)

    def fit(trainer, *args, **kwargs):
        nonlocal stage
        stage = 'training'
        return original_fit(trainer, *args, **kwargs)

    def predict(trainer, *args, **kwargs):
        nonlocal stage
        stage = ('reload_and_predict'
                 if id(trainer) in loaded else 'prediction')
        return original_predict(trainer, *args, **kwargs)

    def load(*args, **kwargs):
        if 'accelerator' in inspect.signature(original_load).parameters:
            kwargs['accelerator'] = 'gpu'
        trainer = original_load(*args, **kwargs)
        loaded.add(id(trainer))
        return trainer

    with (ExitStack() as
          stack, patch.dict(os.environ, {'KUMO_ACCELERATOR': 'gpu'}),
          patch.object(CoreTrainer, 'create_pl_trainer', create),
          patch.object(kumo.Trainer, '__init__',
                       initialize), patch.object(kumo.Trainer, 'fit', fit),
          patch.object(kumo.Trainer, 'predict', predict),
          patch.object(kumo.Trainer, 'load', staticmethod(load)),
          patch.object(task_checks, 'verify_reload', measured_reload),
          patch.object(runpy, 'run_path', load_checks), retries):
        evidence['batch_retries_disabled'] = strict_batches
        yield


def run_prepared_group(args, output, report):
    import torch
    import wheel_learning as binary
    import wheel_scale_data as data
    import wheel_scale_learning as learning
    import wheel_task_checks as checks

    tasks = [
        name for name in data.CASES
        if args.group == 'all' or (name == 'binary') == (args.group == 'binary')
    ]
    seeds = list(SEED_PROFILES[args.seed_profile])
    report.update(group=args.group, profiles={}, seeds=seeds,
                  release_approved=False)
    report['configuration']['seed_profile'] = args.seed_profile
    report['configuration']['scale_bench_rung'] = args.scale_bench_rung
    if getattr(args, 'ranked_links', False):
        report['configuration']['ranked_links'] = True
    if args.calibration_budget is not None:
        report['calibration_fault'] = dict(kind='reduced_training_budget',
                                           budget=args.calibration_budget)
        for task in tasks:
            checks.weakened_training_plan(
                learning.make_plan(task, 0).as_json(), args.calibration_budget)
    result_file = output / 'scale-result.json'
    for task in tasks:
        ranked_options = ({
            'ranked_links': True
        } if getattr(args, 'ranked_links', False) and task.endswith('link') else
                          {})
        profile = args.scale_bench_data / data.CASES[task][
            'graph'] / args.scale_bench_rung
        normal = data.prepare_profile(profile, args.scale_bench_source, task,
                                      output / 'prepared' / task / 'normal',
                                      **ranked_options)
        report['profiles'][task] = normal['identity']
        for seed in seeds:
            for independent in ((False, ) if args.seed_profile == 'pr' else
                                (False, True)):
                case = 'independent' if independent else 'learnable'
                name = f'{task}-{seed}-{case}'
                report['stage'] = name
                print(name, flush=True)
                try:
                    prepared = normal
                    if independent:
                        directory = output / 'prepared' / task / case / str(
                            seed)
                        if (args.control_fixtures is not None and task
                                in ('multilabel', 'multilabel_ranking')):
                            prepared = data.load_control(
                                normal, seed,
                                args.control_fixtures / task / f'seed-{seed}',
                                directory)
                        elif task == 'static_link':
                            prepared = data.prepare_profile(
                                profile, args.scale_bench_source, task,
                                directory, control_seed=seed, **ranked_options)
                        else:
                            prepared = data.control_table(
                                normal, seed, directory)
                    row = learning.run_case(
                        prepared, task, seed, independent, output / name,
                        calibration_budget=args.calibration_budget)
                    if 'frozen_control' in prepared:
                        row['frozen_control'] = prepared['frozen_control']
                    report['runs'].append(row)
                finally:
                    result_file.write_text(
                        json.dumps(report, indent=2, allow_nan=False))
                    gc.collect()
                    torch.cuda.empty_cache()
    report['status'] = 'MEASURED'
    verify_prepared_report(
        report, calibration_dir=(output if args.calibration_budget is not None
                                 else None))
    if args.seed_profile == 'pr':
        report['learning_quality_policy'] = 'report_only'
        return
    report['profile_learning'] = learning.validate_profile_learning(
        report['runs'], calibration=args.calibration_budget is not None,
        seeds=seeds)
    binary_rows = [row for row in report['runs'] if row['task'] == 'binary']
    if binary_rows:
        report['binary_control'] = binary.chance_sanity(
            [dict(runs=binary_rows)], seeds)
    additional = [row for row in report['runs'] if row['task'] != 'binary']
    if additional:
        report['task_controls'] = checks.independent_controls([additional])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--group', choices=('binary', 'additional', 'all'),
                        required=True)
    parser.add_argument('--device', choices=('cuda', ), required=True)
    parser.add_argument('--scale-bench-data', type=Path)
    parser.add_argument('--scale-bench-source', type=Path)
    parser.add_argument('--scale-bench-rung', choices=('pr', '100M'),
                        help='Prepared dataset profile (default: pr)')
    parser.add_argument(
        '--seed-profile', choices=('pr', 'full'), default='full',
        help='One PR correctness seed or five nightly/release seeds')
    parser.add_argument('--control-fixtures', type=Path)
    parser.add_argument(
        '--ranked-links', action='store_true',
        help='Stream top-K link scores instead of dense catalogue matrices')
    parser.add_argument('--dependency-profile', type=Path)
    parser.add_argument('--calibration-budget', type=int, nargs=2,
                        metavar=('EPOCHS', 'STEPS'),
                        help='Measure weaker prepared training; never release')
    args = parser.parse_args()
    if bool(args.scale_bench_data) != bool(args.scale_bench_source):
        parser.error('data and source must be provided together')
    if args.group == 'all' and args.scale_bench_data is None:
        parser.error('all requires prepared profiles')
    prepared = args.scale_bench_data is not None
    if args.ranked_links and not prepared:
        parser.error('ranked links require prepared profiles')
    if args.seed_profile != 'full' and not prepared:
        parser.error('PR seed profile requires prepared profiles')
    if args.scale_bench_rung is not None and not prepared:
        parser.error('rung requires prepared profiles')
    args.scale_bench_rung = args.scale_bench_rung or 'pr'
    if args.calibration_budget is not None and not prepared:
        parser.error('calibration requires prepared profiles')
    if args.control_fixtures is not None and not prepared:
        parser.error('control fixtures require prepared profiles')
    if prepared and args.dependency_profile is None:
        parser.error('prepared profiles require frozen dependencies')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    additional = args.group == 'additional'
    sys.path.insert(0, str(ROOT))
    worker = None if prepared else runpy.run_path(
        str(ROOT /
            ('wheel_task_learning.py' if additional else 'wheel_learning.py')))
    report = dict(
        status='RUNNING', run_id=uuid.uuid4().hex,
        source_sha256=(source_identity(
            prepared=prepared) if additional or prepared else hashlib.sha256(
                (ROOT / 'wheel_learning.py').read_bytes()).hexdigest()),
        gpu_source_sha256=source_identity(prepared=prepared),
        python=sys.version, platform=platform.platform(), seeds=list(range(5)),
        runs=[])
    result_file = output / (
        'scale-result.json' if prepared else
        'task-result.json' if additional else 'learning-result.json')
    started = time.monotonic()
    try:
        import torch

        import kumo.engine as kumo

        if (sys.platform != 'linux' or platform.machine() != 'x86_64'
                or sys.version_info[:2] != (3, 12)
                or importlib.metadata.version('kumo-engine') != args.version
                or kumo.__version__ != args.version
                or not Path(kumo.__file__).resolve().is_relative_to(
                    Path(sys.prefix)) or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1):
            raise ValueError(
                'Use the isolated Linux x64 Python 3.12 CUDA wheel')
        torch.set_num_threads(4)
        sys.path.insert(0, str(ROOT))
        binary = runpy.run_path(str(ROOT / 'wheel_learning.py'))
        packages = {
            re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower(): dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata['Name'].lower().replace('_', '-') != 'kumo-engine'
        }
        if args.dependency_profile is not None:
            verify_dependencies(packages, args.dependency_profile)
        report.update(
            version=args.version, import_path=kumo.__file__,
            dependencies_sha256=binary['fingerprint'](packages),
            dependencies=packages, configuration=dict(
                os=sys.platform, machine=platform.machine(), python='3.12',
                device='cuda', threads=4, gpu=torch.cuda.get_device_name(0),
                cuda=torch.version.cuda, torch=torch.__version__))
        if not additional and not prepared:
            report['settings'] = binary['SETTINGS']
        if prepared:
            run_prepared_group(args, output, report)
            return 0
        node = runpy.run_path(str(
            ROOT / 'wheel_node_tasks.py')) if additional else None
        temporal = runpy.run_path(str(
            ROOT / 'wheel_temporal_tasks.py')) if additional else None
        tasks = ([
            'static_link',
            *(name for name in worker['TASKS'] if name != 'static_link')
        ] if additional else ['binary'])
        for task in tasks:
            for seed in range(5):
                for independent in (False, True):
                    case = 'independent' if independent else 'learnable'
                    name = (f'{task}-{seed}-{case}'
                            if additional else f'seed-{seed}-{case}')
                    report['stage'] = name
                    print(name, flush=True)
                    evidence = {}
                    try:
                        with observe_gpu(evidence):
                            if additional:
                                runner = node if task in node[
                                    'TASKS'] else temporal
                                row = runner['run_case'](task, seed,
                                                         independent,
                                                         output / name)
                                row.update(task=task, seed=seed, case=case,
                                           status='MEASURED')
                                row['score'] = worker['checks']()['score'](
                                    row['metric'], row['targets'],
                                    row['predictions'])
                                row['baseline_score'] = worker['checks'](
                                )['score'](row['metric'], row['targets'],
                                           row['baseline_predictions'])
                            else:
                                row = binary['run_case'](
                                    Path.cwd() / 'learning-work' / name,
                                    output / name, seed, independent)
                        verify_devices(evidence,
                                       ('training', 'prediction',
                                        'reload_and_predict') if additional else
                                       ('training', ))
                        row.update(evidence)
                        report['runs'].append(row)
                    finally:
                        (output / f'{name}-device.json').write_text(
                            json.dumps(evidence, indent=2))
                        result_file.write_text(json.dumps(report, indent=2))
                    gc.collect()
                    torch.cuda.empty_cache()
        report['status'] = 'MEASURED'
        verify_report(report, additional=additional)
        if additional:
            worker['validate_report'](report, list(range(5)))
        else:
            report['mean_auroc'] = binary['summarize'](report['runs'],
                                                       list(range(5)))
            report['chance_sanity'] = binary['chance_sanity']([report],
                                                              list(range(5)))
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        result_file.write_text(json.dumps(report, indent=2))
    return 0 if report['status'] == 'MEASURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
