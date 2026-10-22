#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report provisional quality limits while rejecting incompatible evidence."""

import argparse
import json
import math
import runpy
import statistics
from pathlib import Path

WORKER = runpy.run_path(str(Path(__file__).with_name('wheel_learning.py')))


def report_comparison(reference, candidate, baseline):
    limits = baseline['limits']
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
            if report.get(field) != baseline[field]:
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
    for row in sorted(after['runs'], key=lambda row:
                      (row['seed'], row['case'])):
        previous = old[row['seed'], row['case']]
        for field in ('training_seed', 'data_sha256', 'plan_sha256'):
            if previous[field] != row[field]:
                raise ValueError(f'Unmatched learning runs: {field}')
        if row['case'] == 'learnable':
            drop = previous['auroc'] - row['auroc']
            drops.append(
                dict(seed=row['seed'], reference=previous['auroc'],
                     candidate=row['auroc'], drop=drop))
            check('candidate', 'max_seed_drop', drop, seed=row['seed'])
    mean_drop = statistics.mean(row['drop'] for row in drops)
    check('candidate', 'max_mean_drop', mean_drop)
    return dict(
        status='MEASURED', quality_status='FAIL' if failures else 'PASS',
        mean_auroc=means, mean_auroc_drop=mean_drop, per_seed_auroc_drop=drops,
        limits=limits, failed_limits=failures,
        chance_sanity=WORKER['chance_sanity']([before, after],
                                              baseline['seeds']),
        release_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = report_comparison(*json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
