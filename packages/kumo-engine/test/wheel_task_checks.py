# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate heldout task scores, reloads, and independent-label controls."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True,
                   allow_nan=False).encode()).hexdigest()


def apply_training_budget(plan, budget):
    """Record and reduce the actual plan only for sensitivity measurements."""
    if budget is None:
        return None
    original_plan = plan.as_json()
    reduced = weakened_training_plan(original_plan, budget)['optimization']
    optimization = plan.optimization
    optimization.max_epochs = reduced['max_epochs']
    optimization.min_steps_per_epoch = reduced['min_steps_per_epoch']
    optimization.max_steps_per_epoch = reduced['max_steps_per_epoch']
    return dict(kind='reduced_training_budget', budget=budget[:],
                original_plan=original_plan)


def weakened_training_plan(original, budget):
    if (not isinstance(budget, list) or len(budget) != 2
            or any(type(value) is not int or value < 1 for value in budget)):
        raise ValueError('Calibration needs positive integer epochs and steps')
    fields = ('max_epochs', 'min_steps_per_epoch', 'max_steps_per_epoch')
    try:
        before = [original['optimization'][key] for key in fields]
    except (KeyError, TypeError) as error:
        raise ValueError(
            'Missing original calibration training plan') from error
    reduced = [budget[0], budget[1], budget[1]]
    if (any(type(value) is not int or value < 1 for value in before)
            or any(new > old for new, old in zip(reduced, before))
            or reduced == before):
        raise ValueError('Calibration must reduce the normal training budget')
    result = copy.deepcopy(original)
    result['optimization'].update(zip(fields, reduced))
    return result


def verify_calibration_plans(reference, candidate, reference_dir,
                             candidate_dir):
    filename = ('selected-plan.json' if candidate['task']
                in ('forecasting', 'static_link',
                    'temporal_link') else 'selected-model-plan.json')
    plans = []
    verify = runpy.run_path(str(
        Path(__file__).with_name('wheel_learning.py')))['verify_plan']
    for row, directory in ((reference, reference_dir), (candidate,
                                                        candidate_dir)):
        requested = json.loads(
            (Path(directory) / 'requested-plan.json').read_text())
        selected = json.loads((Path(directory) / filename).read_text())
        if fingerprint(selected) != row['plan_sha256']:
            raise ValueError('Actual calibration plan differs from its hash')
        verify(requested, selected)
        plans.append((requested, selected))
    fault = candidate['calibration_fault']
    if (fault['original_plan'] != plans[0][0] or plans[1][0]
            != weakened_training_plan(plans[0][0], fault['budget'])
            or plans[1][1] != weakened_training_plan(plans[0][1],
                                                     fault['budget'])):
        raise ValueError(
            'A calibration change is not the declared training budget')
    return dict(task=candidate['task'], seed=candidate['seed'],
                case=candidate['case'],
                reference_plan_sha256=reference['plan_sha256'],
                candidate_plan_sha256=candidate['plan_sha256'])


def _matrix(values):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Expected a finite numerical matrix') from error
    if (result.ndim != 2 or not all(result.shape)
            or not np.isfinite(result).all()):
        raise ValueError('Expected a nonempty finite numerical matrix')
    return result


def _pair(metric, targets, predictions):
    targets, predictions = _matrix(targets), _matrix(predictions)
    if targets.shape != predictions.shape:
        raise ValueError(
            'Targets and full prediction scores must have equal shapes')
    if metric not in ('accuracy', 'ap_macro', 'recall@5', 'mae'):
        raise ValueError(f'Unsupported metric: {metric}')
    if metric != 'mae' and not np.isin(targets, [0, 1]).all():
        raise ValueError('Classification and ranking labels must be binary')
    if metric == 'accuracy' and (targets.shape[1] < 2
                                 or not (targets.sum(axis=1) == 1).all()):
        raise ValueError('Accuracy needs one-hot targets for every class')
    if metric == 'ap_macro' and not (
        (targets.sum(axis=0) > 0)
            & (targets.sum(axis=0) < len(targets))).all():
        raise ValueError('Average precision needs positive and negative labels')
    if metric == 'recall@5' and (targets.shape[1] < 5
                                 or not (targets.sum(axis=1) > 0).all()):
        raise ValueError('Recall needs a full catalog and nonempty target sets')
    return targets, predictions


def _score(metric, targets, predictions):
    if metric == 'accuracy':
        return float(
            (targets.argmax(axis=1) == predictions.argmax(axis=1)).mean())
    if metric == 'ap_macro':
        return float(
            average_precision_score(targets, predictions, average='macro'))
    if metric == 'recall@5':
        selected = np.argsort(-predictions, axis=1, kind='stable')[:, :5]
        hits = np.take_along_axis(targets, selected, axis=1).sum(axis=1)
        return float((hits / targets.sum(axis=1)).mean())
    return float(np.abs(targets - predictions).mean())


def score(metric, targets, predictions):
    result = _score(metric, *_pair(metric, targets, predictions))
    if not np.isfinite(result):
        raise ValueError('The metric is not finite')
    return result


def verify_predictions(expected_ids, actual_ids, predictions):
    values = _matrix(predictions)
    try:
        matched = (list(expected_ids) == list(actual_ids)
                   and len(set(expected_ids)) == len(expected_ids)
                   and len(expected_ids) == values.shape[0])
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid prediction entity IDs') from error
    if not matched:
        raise ValueError('Prediction entity IDs must match once and in order')
    return values


def verify_reload(expected, actual):
    expected, actual = _matrix(expected), _matrix(actual)
    if (expected.shape != actual.shape
            or not np.allclose(expected, actual, atol=1e-6, rtol=1e-6)):
        raise ValueError('Reloaded prediction scores differ')
    return float(np.max(np.abs(expected - actual)))


def independent_controls(arms):
    """Test each task with shared row-label shuffles across matched wheel arms."""
    if not arms:
        raise ValueError('Missing independent-label arms')
    indexed = []
    for arm in arms:
        rows = [row for row in arm if row['case'] == 'independent']
        keys = [(row['task'], row['seed']) for row in rows]
        if not rows or len(set(keys)) != len(rows):
            raise ValueError('Missing or duplicate independent-label runs')
        indexed.append(dict(zip(keys, rows)))
    keys = set(indexed[0])
    if any(set(arm) != keys for arm in indexed):
        raise ValueError('Unmatched independent-label runs')
    results = {}
    for task in sorted({key[0] for key in keys}):
        selected = sorted(key for key in keys if key[0] == task)
        first = indexed[0][selected[0]]
        metric = first['metric']
        conditional = 'control_eligible' in first
        observed = np.zeros(len(arms))
        null = np.zeros((999, len(arms)))
        rng = np.random.default_rng(0)
        for key in selected:
            rows = [arm[key] for arm in indexed]
            validated = [
                _pair(row['metric'], row['targets'], row['predictions'])
                for row in rows
            ]
            targets = validated[0][0]
            if any(row['metric'] != metric
                   or not np.array_equal(labels, targets)
                   for row, (labels, _) in zip(rows, validated)):
                raise ValueError(
                    'Unmatched independent-label targets or metric')
            eligible = None
            if conditional or any('control_eligible' in row for row in rows):
                masks = [
                    np.asarray(row.get('control_eligible')) for row in rows
                ]
                eligible = masks[0]
                if (not conditional or metric != 'recall@5'
                        or any(mask.dtype != bool or mask.shape != targets.shape
                               or not np.array_equal(mask, eligible)
                               for mask in masks)
                        or np.any(targets[~eligible] != 0)):
                    raise ValueError(
                        'Invalid or unmatched eligible target masks')
            direction = -1 if metric == 'mae' else 1
            for index, (_, predictions) in enumerate(validated):
                observed[index] += direction * _score(
                    metric, targets, predictions) / len(selected)
            for draw in range(999):
                if eligible is None:
                    shuffled = targets[rng.permutation(len(targets))]
                else:
                    order = np.argsort(
                        np.where(eligible, rng.random(targets.shape), np.inf),
                        axis=1)
                    shuffled = np.zeros_like(targets)
                    np.put_along_axis(
                        shuffled, order,
                        np.arange(targets.shape[1])[None, :]
                        < targets.sum(axis=1)[:, None], axis=1)
                for index, (_, predictions) in enumerate(validated):
                    null[draw, index] += direction * _score(
                        metric, shuffled, predictions) / len(selected)
        statistic = float(observed.max())
        p_value = (1 + int(
            (null.max(axis=1) >= statistic - 1e-12).sum())) / 1000
        alpha = .01 / 7
        if p_value <= alpha:
            raise ValueError(
                f'{task}: independent labels show suspicious learning')
        results[task] = dict(
            method='joint_monte_carlo_label_permutation', draws=999, seed=0,
            alpha=alpha, statistic=statistic, p_value=p_value,
            null_method=('uniform_eligible_items'
                         if conditional else 'shared_label_rows'))
    return results
