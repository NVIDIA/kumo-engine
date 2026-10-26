#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare learning scores and retain historical limits as diagnostics."""

import argparse
import json
import math
import runpy
from fractions import Fraction
from pathlib import Path

WORKER = runpy.run_path(str(Path(__file__).with_name('wheel_learning.py')))


def report_comparison(reference, candidate, baseline):
    gpu = reference['learning'].get('configuration', {}).get('device') == 'cuda'
    if gpu:
        gpu_worker = runpy.run_path(
            str(Path(__file__).with_name('wheel_gpu_learning.py')))
        for arm in (reference, candidate):
            gpu_worker['verify_report'](arm['learning'], additional=False)
    limits = baseline['limits']
    alert_limit = baseline.get('release_max_mean_drop', limits['max_mean_drop'])
    if (type(alert_limit) not in (int, float) or not math.isfinite(alert_limit)
            or not 0 <= alert_limit <= 1):
        raise ValueError('Invalid reporting mean-drop limit')
    if any(not math.isfinite(limits[key]) or not 0 <= limits[key] <= 1
           for key in ('learnable_min', 'independent_distance', 'max_mean_drop',
                       'max_seed_drop')):
        raise ValueError('Invalid learning limit')
    if limits['learnable_min'] <= .5 + limits['independent_distance']:
        raise ValueError(
            'Learning limits must separate signal from independent labels')
    if baseline.get('validation', {}).get('status') != 'PASS':
        raise ValueError('Reference calibration needs passing reserved runs')
    if reference['wheel'] != baseline['reference_wheel']:
        raise ValueError('Reference wheel differs from the quality baseline')
    means = {}
    failures = []

    def check(arm, name, actual, *, minimum=False, seed=None):
        limit = limits[name]
        failed = actual < limit - 1e-12 if minimum else actual > limit + 1e-12
        if failed:
            failures.append(
                dict(arm=arm, check=name, actual=actual, limit=limit,
                     **({
                         'seed': seed
                     } if seed is not None else {})))

    for name, result in (('reference', reference), ('candidate', candidate)):
        report = result['learning']
        if result.get('status') != 'MEASURED' or report.get(
                'status') != 'MEASURED':
            raise ValueError('Both wheels need completed learning measurements')
        for field in ('source_sha256', 'settings', 'configuration',
                      'dependencies_sha256', 'seeds'):
            expected = (reference['learning'][field] if gpu
                        and field == 'configuration' else baseline[field])
            if report.get(field) != expected:
                raise ValueError(
                    f'Learning evidence differs from baseline: {field}')
        means[name] = WORKER['summarize'](report['runs'], report['seeds'])
        check(name, 'learnable_min', means[name]['learnable'], minimum=True)
        check(name, 'independent_distance',
              abs(means[name]['independent'] - .5))
    before, after = reference['learning'], candidate['learning']
    for field in ('python', 'platform'):
        if before.get(field) != after.get(field):
            raise ValueError(f'Wheel environments differ: {field}')
    old = {(row['seed'], row['case']): row for row in before['runs']}
    drops = []
    exact_drops = []
    for row in sorted(after['runs'], key=lambda row:
                      (row['seed'], row['case'])):
        previous = old[row['seed'], row['case']]
        for field in ('training_seed', 'data_sha256', 'plan_sha256'):
            if previous[field] != row[field]:
                raise ValueError(f'Unmatched learning runs: {field}')
        if row['case'] == 'learnable':
            drop = (Fraction(str(previous['auroc'])) -
                    Fraction(str(row['auroc'])))
            exact_drops.append(drop)
            drops.append(
                dict(seed=row['seed'], reference=previous['auroc'],
                     candidate=row['auroc'], drop=float(drop)))
            check('candidate', 'max_seed_drop', float(drop), seed=row['seed'])
    mean_drop = sum(exact_drops) / len(exact_drops)
    check('candidate', 'max_mean_drop', float(mean_drop))
    return dict(
        status='MEASURED', quality_status='FAIL' if failures else 'PASS',
        mean_auroc=means, mean_auroc_drop=float(mean_drop),
        per_seed_auroc_drop=drops,
        mean_drop_alert=dict(limit=alert_limit, exceeded=mean_drop
                             > Fraction(str(alert_limit))), limits=limits,
        failed_limits=failures, chance_sanity=WORKER['chance_sanity'](
            [before, after], baseline['seeds']), release_approved=False)


def compare_release(reference, candidate, baseline):
    if any(arm['learning'].get('configuration', {}).get('device') == 'cuda'
           for arm in (reference, candidate)):
        raise ValueError(
            'GPU limits require separate qualification before release')
    seeds = baseline['seeds']
    if (not isinstance(seeds, list) or len(seeds) != 5
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != 5):
        raise ValueError(
            'Release comparison requires five distinct integer seeds')
    if 'release_max_mean_drop' not in baseline:
        raise ValueError('Release mean-drop limit is required')
    seen = set()
    for result in (reference, candidate):
        WORKER['verify_unique_evidence'](result, seen)
        report = result['learning']
        if any(type(seed) is not int for seed in report['seeds']):
            raise ValueError('Training seeds must be integers')
        for row in report['runs']:
            if (type(row['seed']) is not int
                    or type(row['training_seed']) is not int
                    or row['training_seed'] != row['seed']):
                raise ValueError(
                    'Training seed differs from the requested seed')
    result = report_comparison(reference, candidate, baseline)
    passed = not result['mean_drop_alert']['exceeded']
    result.update(status='PASS' if passed else 'FAIL', release_approved=passed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--release', action='store_true',
                        help='Enforce the approved five-seed mean-drop limit')
    parser.add_argument('--tasks', action='store_true',
                        help='Require evidence for all seven additional tasks')
    parser.add_argument(
        '--calibration-task-dirs', nargs=2, type=Path,
        metavar=('REFERENCE', 'CANDIDATE'),
        help='Audit actual weaker-training plans; never release')
    args = parser.parse_args()
    if args.calibration_task_dirs is not None and (args.release
                                                   or not args.tasks):
        parser.error(
            'Task calibration cannot approve release and requires --tasks')
    compare = compare_release if args.release else report_comparison
    reference, candidate, baseline = json.loads(args.input.read_text())
    result = compare(reference, candidate, baseline)
    if args.tasks:
        if any('task_learning' not in arm for arm in (reference, candidate)):
            raise ValueError('Additional learning results are required')
        tasks = runpy.run_path(
            str(Path(__file__).with_name('wheel_task_learning.py')))
        gpu = reference['learning'].get('configuration',
                                        {}).get('device') == 'cuda'
        gpu_worker = (runpy.run_path(
            str(Path(__file__).with_name('wheel_gpu_learning.py')))
                      if gpu else None)
        for arm in (reference, candidate):
            if (arm['task_learning'].get('source_sha256')
                    != (gpu_worker['source_identity']()
                        if gpu else tasks['source_identity']())
                    or arm['task_learning'].get('dependencies_sha256')
                    != arm['learning']['dependencies_sha256']):
                raise ValueError(
                    'Additional task source or dependencies differ')
            if gpu:
                gpu_worker['verify_report'](arm['task_learning'],
                                            additional=True)
                if arm['task_learning']['configuration'] != reference[
                        'learning']['configuration']:
                    raise ValueError('GPU learning task runtimes differ')
        comparison = tasks['compare_tasks'](
            reference['task_learning'], candidate['task_learning'],
            calibration_dirs=args.calibration_task_dirs)
        comparison.update(
            source_sha256=reference['task_learning']['source_sha256'],
            dependencies_sha256=reference['task_learning']
            ['dependencies_sha256'], reference_wheel=reference['wheel'],
            seeds=reference['task_learning']['seeds'])
        quality = runpy.run_path(
            str(Path(__file__).with_name('wheel_task_quality.py')))
        profile = json.loads(
            Path(__file__).with_name('wheel_task_quality.json').read_text())
        if gpu:
            consistency = {}
            for stage in ('prediction', 'reload_and_predict'):
                measured = [
                    m for arm in (reference, candidate)
                    for row in arm['task_learning']['runs']
                    for m in row['consistency_measurements']
                    if m['stage'] == stage
                ]
                consistency[stage] = dict(
                    comparisons=len(measured),
                    max_abs_difference=max(m['max_abs_difference']
                                           for m in measured),
                    cpu_tolerance_exceeded=sum(m['cpu_tolerance_exceeded']
                                               for m in measured))
            comparison.update(
                quality_status='UNVALIDATED_GPU',
                configuration=reference['learning']['configuration'],
                numerical_consistency=dict(status='MEASURED', qualified=False,
                                           gpu_tolerance=None,
                                           stages=consistency))
            result['additional_tasks'] = comparison
        else:
            result['additional_tasks'] = quality['assess_quality'](comparison,
                                                                   profile)
        if args.release and not result['additional_tasks'][
                'quality_gate_approved']:
            result.update(status='FAIL', release_approved=False)
    args.output.write_text(json.dumps(result, indent=2))
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
