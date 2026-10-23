# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply task-specific mean-drop limits only after fresh qualification."""

import hashlib
import json
import math
import re
import runpy
from datetime import datetime
from fractions import Fraction
from pathlib import Path

MEASUREMENT = runpy.run_path(
    str(Path(__file__).with_name('wheel_task_learning.py')))
TASKS = MEASUREMENT['TASKS']
RELIABILITY = dict(joint_false_failure_max=.05, task_detection_min=.95,
                   simultaneous_confidence=.95, claims=8, reserved_trials=99)


def number(value, *, limit=False):
    if limit and isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ArithmeticError) as error:
            raise ValueError('Invalid exact task quality limit') from error
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Task quality needs finite numerical values')
    return Fraction(str(value))


def scores(tasks):
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise ValueError('All seven task metrics are required')
    details, drops = {}, {}
    for name, (metric, _) in TASKS.items():
        task = tasks[name]
        rows = task.get('per_seed', [])
        if (task.get('metric') != metric or len(rows) != 5
                or {row['seed']
                    for row in rows} != set(range(5))
                or any(type(row['seed']) is not int for row in rows)):
            raise ValueError(
                'Task quality requires five matching integer seeds')
        before = [number(row['reference']) for row in rows]
        after = [number(row['candidate']) for row in rows]
        baseline = [number(row['baseline']) for row in rows]
        if any(value < 0 or (metric != 'mae' and value > 1)
               for value in before + after + baseline):
            raise ValueError('Task quality score is outside its metric range')
        direction = -1 if metric == 'mae' else 1
        mean_before, mean_after = sum(before) / 5, sum(after) / 5
        drops[name] = direction * (mean_before - mean_after)
        details[name] = dict(
            task, mean_reference=float(mean_before),
            mean_candidate=float(mean_after),
            mean_quality_drop=float(drops[name]),
            higher_is_better=direction == 1,
            metric_units='target_units' if metric == 'mae' else 'fraction',
            per_seed=[
                dict(row, quality_drop=float(direction * (old - new)))
                for row, old, new in zip(rows, before, after)
            ])
    return details, drops


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.utcoffset() is None:
            raise ValueError('A timezone is required')
        return result
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            'Qualification needs dated calibration and validation') from error


def validate_profile(profile, comparison):
    if profile.get('status') != 'VALIDATED' or profile.get(
            'reliability') != RELIABILITY:
        raise ValueError(
            'The declared reliability policy must pass before release')
    for key in ('source_sha256', 'dependencies_sha256', 'reference_wheel'):
        if profile.get(key) != comparison.get(key):
            raise ValueError(
                f'Quality qualification environment differs: {key}')
    if profile['source_sha256'] != MEASUREMENT['source_identity']():
        raise ValueError(
            'Quality qualification uses a different measurement source')
    for key in ('source_sha256', 'dependencies_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', profile.get(key, '')):
            raise ValueError(
                'Quality qualification needs source and dependency hashes')
    if (profile.get('seeds') != list(range(5))
            or any(type(seed) is not int for seed in profile['seeds'])
            or comparison.get('seeds') != profile['seeds']):
        raise ValueError('Quality qualification needs five matching seeds')
    for key in ('limits', 'minimum_losses', 'rationales'):
        if not isinstance(profile.get(key), dict) or set(
                profile[key]) != set(TASKS):
            raise ValueError(f'All seven quality {key} are required')
    limits = {
        name: number(profile['limits'][name], limit=True)
        for name in TASKS
    }
    minimums = {name: number(profile['minimum_losses'][name]) for name in TASKS}
    for name, (metric, _) in TASKS.items():
        if (not 0 <= limits[name] < minimums[name]
                or (metric != 'mae' and minimums[name] > 1)
                or not isinstance(profile['rationales'][name], str)
                or not profile['rationales'][name].strip()):
            raise ValueError(
                'Every limit needs a larger justified blockworthy loss')
    frozen = timestamp(profile.get('frozen_at'))
    seen_ids, seen_runs, seen_hashes = set(), set(), set()
    measured = {}
    for phase in ('calibration', 'validation'):
        groups = dict(unchanged=[], weakened=[])
        if not isinstance(profile.get(phase), list):
            raise ValueError(f'Missing {phase} trials')
        for trial in profile[phase]:
            kind = trial.get('kind')
            if (kind not in groups or trial.get('status') != 'PASS'
                    or trial.get('audit_passed') is not True):
                raise ValueError('Incomplete or unaudited qualification trial')
            identity, digest = trial.get('id'), trial.get('evidence_sha256')
            run_ids = [
                trial.get('reference_run_id'),
                trial.get('candidate_run_id')
            ]
            if (not isinstance(identity, str) or not identity
                    or identity in seen_ids or not isinstance(digest, str)
                    or not re.fullmatch('[0-9a-f]{64}', digest)
                    or digest in seen_hashes or any(
                        not isinstance(run, str) or not run or run in seen_runs
                        for run in run_ids) or run_ids[0] == run_ids[1]):
                raise ValueError(
                    'Qualification trials and models must be fresh')
            seen_ids.add(identity)
            seen_hashes.add(digest)
            seen_runs.update(run_ids)
            for key in ('source_sha256', 'dependencies_sha256',
                        'reference_wheel', 'seeds'):
                if trial.get(key) != profile[key]:
                    raise ValueError(
                        f'Qualification trial environment differs: {key}')
            if (any(type(seed) is not int for seed in trial['seeds']) or
                    trial.get('candidate_wheel') != profile['reference_wheel']):
                raise ValueError(
                    'Qualification requires unchanged wheels and matching seeds'
                )
            started, completed = timestamp(trial.get('started_at')), timestamp(
                trial.get('completed_at'))
            if (completed < started
                    or (phase == 'calibration' and completed > frozen)
                    or (phase == 'validation' and started <= frozen)):
                raise ValueError(
                    'Reserved trials must start after limits are frozen')
            fault = trial.get('fault')
            if kind == 'weakened':
                if (not isinstance(fault, dict)
                        or fault != profile.get('frozen_fault')
                        or fault.get('kind') != 'reduced_training_budget'
                        or not isinstance(fault.get('budget'), list)
                        or len(fault['budget']) != 2 or any(
                            type(value) is not int or value < 1
                            for value in fault['budget'])):
                    raise ValueError(
                        'Known regressions need actual weaker training')
            elif fault is not None:
                raise ValueError(
                    'An unchanged trial cannot contain weaker training')
            _, drops = scores(trial['tasks'])
            groups[kind].append(drops)
        measured[phase] = groups
    calibration = measured['calibration']
    if any(len(group) < 6 for group in calibration.values()):
        raise ValueError(
            'Repeated normal and weaker calibration trials are required')
    for name in TASKS:
        noise = max(abs(trial[name]) for trial in calibration['unchanged'])
        loss = min(trial[name] for trial in calibration['weakened'])
        if limits[name] != noise or loss < minimums[name]:
            raise ValueError(
                'Limits must equal measured variation and separate the known loss'
            )
    validation = measured['validation']
    count = RELIABILITY['reserved_trials']
    if any(len(group) != count for group in validation.values()):
        raise ValueError('The fixed reserved trial count is incomplete')
    if any(
            any(trial[name] > limits[name] for name in TASKS)
            for trial in validation['unchanged']):
        raise ValueError('Reserved unchanged-wheel validation failed')
    if any(
            any(trial[name] < minimums[name] or trial[name] <= limits[name]
                for name in TASKS) for trial in validation['weakened']):
        raise ValueError('Reserved known-regression validation failed')
    alpha = (1 - RELIABILITY['simultaneous_confidence']) / RELIABILITY['claims']
    lower = alpha**(1 / count)
    if (1 - lower > RELIABILITY['joint_false_failure_max']
            or lower < RELIABILITY['task_detection_min']):
        raise ValueError(
            'Fresh trials do not establish the declared reliability')
    return dict(status='PASS', unchanged_trials=count, weakened_trials=count,
                false_failure_upper_bound=1 - lower,
                detection_lower_bound=lower, simultaneous_confidence=.95)


def assess_quality(comparison, profile):
    if comparison.get('status') != 'MEASURED' or comparison.get(
            'execution_passed') is not True:
        raise ValueError('Task quality requires completed execution evidence')
    details, drops = scores(comparison.get('tasks'))
    result = dict(
        comparison, tasks=details, quality_gate_approved=False,
        quality_status='UNVALIDATED', quality_limits=None,
        quality_profile_sha256=hashlib.sha256(
            json.dumps(profile, sort_keys=True,
                       allow_nan=False).encode()).hexdigest())
    if 'calibration_fault' in comparison:
        result['quality_status'] = 'CALIBRATION'
        return result
    if profile.get('status') == 'PENDING':
        return result
    qualification = validate_profile(profile, comparison)
    exceeded = []
    for name, task in details.items():
        limit = profile['limits'][name]
        failed = drops[name] > number(limit, limit=True)
        task['mean_drop_alert'] = dict(limit=limit, exceeded=failed)
        if failed:
            exceeded.append(name)
    result.update(quality_status='FAIL' if exceeded else 'PASS',
                  quality_gate_approved=not exceeded,
                  quality_limits=profile['limits'], qualification=qualification)
    return result
