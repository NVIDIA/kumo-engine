#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the seven additional learning tasks without unapproved limits."""

import argparse
import functools
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import runpy
import statistics
import sys
import traceback
import uuid
from pathlib import Path

TASKS = dict(multiclass=('accuracy', 'multiclass_classification'),
             multilabel=('ap_macro', 'multilabel_classification'),
             multilabel_ranking=('recall@5', 'multilabel_ranking'),
             regression=('mae', 'regression'), forecasting=('mae',
                                                            'forecasting'),
             static_link=('recall@5', 'static_link_prediction'),
             temporal_link=('recall@5', 'temporal_link_prediction'))
SEEDS = list(range(5))
REQUIRED_CHECKS = ('fit', 'prediction', 'partial_prediction', 'reload',
                   'hidden_answers')
SOURCES = ('wheel_task_learning.py', 'wheel_task_checks.py',
           'wheel_node_tasks.py', 'wheel_temporal_tasks.py',
           'wheel_learning.py')


@functools.lru_cache(maxsize=1)
def checks():
    return runpy.run_path(str(Path(__file__).with_name('wheel_task_checks.py')))


def source_identity():
    sources = {
        name:
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in SOURCES
    }
    return hashlib.sha256(json.dumps(sources,
                                     sort_keys=True).encode()).hexdigest()


def validate_report(report, seeds, *, calibration=False):
    fault = report.get('calibration_fault')
    if calibration:
        if (not isinstance(fault, dict) or set(fault) != {'kind', 'budget'}
                or fault['kind'] != 'reduced_training_budget'):
            raise ValueError('Missing or invalid calibration fault')
    elif ('calibration_fault' in report
          or any('calibration_fault' in row for row in report.get('runs', []))):
        raise ValueError('A calibration run cannot approve release')
    if (seeds != SEEDS or report.get('seeds') != seeds
            or any(type(seed) is not int for seed in report.get('seeds', []))
            or report.get('status') != 'MEASURED' or not report.get('run_id')):
        raise ValueError('Additional tasks require five completed seed pairs')
    for key in ('python', 'platform', 'run_id'):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ValueError(f'Missing task environment evidence: {key}')
    for key in ('source_sha256', 'dependencies_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', report.get(key, '')):
            raise ValueError(f'Missing task environment evidence: {key}')
    expected = {(task, seed, case)
                for task in TASKS
                for seed in seeds
                for case in ('learnable', 'independent')}
    rows = report['runs']
    if (len(rows) != len(expected) or {(r['task'], r['seed'], r['case'])
                                       for r in rows} != expected):
        raise ValueError(
            'An additional task, seed, or control result is missing')
    if (any(not isinstance(row.get('fit_id'), str) or not row['fit_id'].strip()
            for row in rows) or len({r['fit_id']
                                     for r in rows}) != len(rows)):
        raise ValueError('Additional tasks must use fresh models')
    for row in rows:
        if calibration:
            detail = row.get('calibration_fault')
            if (not isinstance(detail, dict)
                    or set(detail) != {'kind', 'budget', 'original_plan'}
                    or any(detail[key] != fault[key]
                           for key in ('kind', 'budget'))):
                raise ValueError('Missing or inconsistent calibration fault')
            try:
                checks()['weakened_training_plan'](detail['original_plan'],
                                                   fault['budget'])
            except ValueError as error:
                raise ValueError('Invalid calibration training plan') from error
        if (row.get('status') != 'MEASURED' or type(row['seed']) is not int
                or type(row['training_seed']) is not int
                or row['training_seed'] != row['seed']
                or (row['metric'], row['task_type']) != TASKS[row['task']]
                or any(
                    row.get('checks', {}).get(key) != 'PASS'
                    for key in REQUIRED_CHECKS)):
            raise ValueError('An additional task failed execution checks')
        if row['task'] in ('forecasting', 'temporal_link'
                           ) and row['checks'].get('chronology') != 'PASS':
            raise ValueError(
                'Temporal evidence needs passing chronology checks')
        for key in ('data_sha256', 'plan_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row[key]):
                raise ValueError(f'Invalid task evidence: {key}')
        if row['task'] in ('forecasting', 'static_link', 'temporal_link'):
            stages = row.get('stage_sha256')
            if (not isinstance(stages, dict)
                    or set(stages) != {'train', 'val', 'test'}
                    or any(not isinstance(value, str)
                           or not re.fullmatch('[0-9a-f]{64}', value)
                           for value in stages.values())):
                raise ValueError('Invalid task evidence: stage_sha256')
        for key in ('predictions', 'targets', 'baseline_predictions'):
            checks()['verify_predictions'](row['entity_ids'], row['entity_ids'],
                                           row[key])
        if row['task'] == 'forecasting' and len(row['targets'][0]) != 12:
            raise ValueError('Forecasting requires all twelve future steps')
        if row['task'] == 'static_link' and 'control_eligible' not in row:
            raise ValueError('Static links require eligible candidate evidence')
        for key, values in (('score', 'predictions'), ('baseline_score',
                                                       'baseline_predictions')):
            measured = checks()['score'](row['metric'], row['targets'],
                                         row[values])
            if (not math.isfinite(row[key]) or not math.isclose(
                    measured, row[key], abs_tol=1e-12, rel_tol=0)):
                raise ValueError(
                    'Saved task score differs from prediction evidence')


def compare_tasks(reference, candidate, *, calibration_dirs=None):
    validate_report(reference, SEEDS)
    validate_report(candidate, SEEDS, calibration=calibration_dirs is not None)
    if calibration_dirs is not None and len(calibration_dirs) != 2:
        raise ValueError(
            'Both actual calibration evidence directories are required')
    for key in ('source_sha256', 'python', 'platform', 'dependencies_sha256'):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f'Additional task environments differ: {key}')
    if reference['run_id'] == candidate['run_id']:
        raise ValueError('Additional task evidence was reused')
    if ({r['fit_id']
         for r in reference['runs']}
            & {r['fit_id']
               for r in candidate['runs']}):
        raise ValueError('Additional task models were reused between wheels')
    before = {(r['task'], r['seed'], r['case']): r for r in reference['runs']}
    tasks = {
        task: dict(metric=metric, per_seed=[])
        for task, (metric, _) in TASKS.items()
    }
    plan_checks = []
    for row in sorted(candidate['runs'], key=lambda r:
                      (r['task'], r['seed'], r['case'])):
        old = before[row['task'], row['seed'], row['case']]
        for key in ('training_seed', 'data_sha256', 'plan_sha256', 'targets',
                    'entity_ids', 'baseline_predictions'):
            if key == 'plan_sha256' and calibration_dirs is not None:
                name = f'{row["task"]}-{row["seed"]}-{row["case"]}'
                plan_checks.append(checks()['verify_calibration_plans'](
                    old, row, Path(calibration_dirs[0]) / name,
                    Path(calibration_dirs[1]) / name))
                continue
            if old[key] != row[key]:
                raise ValueError(f'Additional task inputs differ: {key}')
        for key in ('control_eligible', 'stage_sha256'):
            if old.get(key) != row.get(key):
                raise ValueError(f'Additional task inputs differ: {key}')
        if row['case'] == 'learnable':
            direction = -1 if row['metric'] == 'mae' else 1
            tasks[row['task']]['per_seed'].append(
                dict(seed=row['seed'], reference=old['score'],
                     candidate=row['score'], baseline=row['baseline_score'],
                     quality_drop=direction * (old['score'] - row['score'])))
    for result in tasks.values():
        result['mean_quality_drop'] = statistics.mean(
            row['quality_drop'] for row in result['per_seed'])
    controls = checks()['independent_controls'](
        [reference['runs'], candidate['runs']])
    result = dict(status='MEASURED', execution_passed=True, tasks=tasks,
                  independent_controls=controls, quality_limits=None,
                  quality_gate_approved=False)
    if calibration_dirs is not None:
        result.update(calibration_fault=candidate['calibration_fault'],
                      calibration_plan_checks=plan_checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--calibration-budget', type=int, nargs=2,
                        metavar=('EPOCHS', 'STEPS'),
                        help='Measure weaker training; never approve release')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', run_id=uuid.uuid4().hex, seeds=SEEDS,
                  source_sha256=source_identity(), python=sys.version,
                  platform=platform.platform(), runs=[])
    if args.calibration_budget is not None:
        report['calibration_fault'] = dict(kind='reduced_training_budget',
                                           budget=args.calibration_budget)
    try:
        import torch

        import kumo.engine as kumo

        if (sys.platform != 'linux' or platform.machine() != 'x86_64'
                or sys.version_info[:2] != (3, 12)
                or importlib.metadata.version('kumo-engine') != kumo.__version__
                or kumo.__version__ != args.version
                or not Path(kumo.__file__).resolve().is_relative_to(
                    Path(sys.prefix)) or torch.cuda.is_available()):
            raise ValueError('Additional tasks require the isolated CPU wheel')
        torch.set_num_threads(4)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        packages = {
            re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower(): dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata['Name'].lower().replace('_', '-') != 'kumo-engine'
        }
        report['dependencies_sha256'] = checks()['fingerprint'](packages)
        node = runpy.run_path(
            str(Path(__file__).with_name('wheel_node_tasks.py')))
        temporal = runpy.run_path(
            str(Path(__file__).with_name('wheel_temporal_tasks.py')))
        for task in TASKS:
            runner = node if task in node['TASKS'] else temporal
            for seed in SEEDS:
                for independent in (False, True):
                    case = 'independent' if independent else 'learnable'
                    name = f'{task}-{seed}-{case}'
                    print(name, flush=True)
                    report['stage'] = name
                    row = runner['run_case'](
                        task, seed, independent, output / name,
                        calibration_budget=args.calibration_budget)
                    row.update(task=task, seed=seed, case=case,
                               status='MEASURED')
                    row['score'] = checks()['score'](row['metric'],
                                                     row['targets'],
                                                     row['predictions'])
                    row['baseline_score'] = checks()['score'](
                        row['metric'], row['targets'],
                        row['baseline_predictions'])
                    report['runs'].append(row)
                    (output / 'task-result.json').write_text(
                        json.dumps(report, indent=2))
                    gc.collect()
        report['status'] = 'MEASURED'
        validate_report(report, SEEDS, calibration=args.calibration_budget
                        is not None)
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        (output / 'task-result.json').write_text(json.dumps(report, indent=2))
    return 0 if report['status'] == 'MEASURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
