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
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def source_identity():
    tasks = runpy.run_path(str(ROOT / 'wheel_task_learning.py'))
    names = (*tasks['SOURCES'], 'wheel_gpu_learning.py')
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


def verify_report(report, *, additional):
    if (report.get('configuration', {}).get('device') != 'cuda'
            or report.get('gpu_source_sha256') != source_identity()):
        raise ValueError('GPU runtime or guard source differs')
    for row in report.get('runs', []):
        verify_devices(row, ('training', 'prediction',
                             'reload_and_predict') if additional else
                       ('training', ))
        if additional:
            measured = row.get('consistency_measurements', [])
            if (len(measured) != 2 or [m.get('stage') for m in measured]
                    != ['prediction', 'reload_and_predict']):
                raise ValueError(
                    'Both GPU consistency measurements are required')
            for item in measured:
                value = item.get('max_abs_difference')
                if (item.get('status') != 'MEASURED'
                        or item.get('qualified') is not False
                        or item.get('gpu_tolerance') is not None
                        or type(value) not in (int, float)
                        or not math.isfinite(value) or value < 0):
                    raise ValueError('Invalid GPU consistency measurement')
    if not report.get('runs'):
        raise ValueError('GPU measurements contain no completed models')


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


@contextmanager
def observe_gpu(evidence):
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

    with (patch.dict(os.environ, {'KUMO_ACCELERATOR': 'gpu'}),
          patch.object(CoreTrainer, 'create_pl_trainer', create),
          patch.object(kumo.Trainer, '__init__',
                       initialize), patch.object(kumo.Trainer, 'fit', fit),
          patch.object(kumo.Trainer, 'predict', predict),
          patch.object(kumo.Trainer, 'load', staticmethod(load)),
          patch.object(task_checks, 'verify_reload', measured_reload),
          patch.object(runpy, 'run_path', load_checks)):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--group', choices=('binary', 'additional'),
                        required=True)
    parser.add_argument('--device', choices=('cuda', ), required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    additional = args.group == 'additional'
    worker = runpy.run_path(
        str(ROOT /
            ('wheel_task_learning.py' if additional else 'wheel_learning.py')))
    report = dict(
        status='RUNNING', run_id=uuid.uuid4().hex,
        source_sha256=(source_identity() if additional else hashlib.sha256(
            (ROOT / 'wheel_learning.py').read_bytes()).hexdigest()),
        gpu_source_sha256=source_identity(), python=sys.version,
        platform=platform.platform(), seeds=list(range(5)), runs=[])
    result_file = output / ('task-result.json'
                            if additional else 'learning-result.json')
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
        report.update(
            version=args.version, import_path=kumo.__file__,
            dependencies_sha256=binary['fingerprint'](packages),
            configuration=dict(os=sys.platform, machine=platform.machine(),
                               python='3.12', device='cuda', threads=4,
                               gpu=torch.cuda.get_device_name(0),
                               cuda=torch.version.cuda,
                               torch=torch.__version__))
        if not additional:
            report['settings'] = binary['SETTINGS']
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
