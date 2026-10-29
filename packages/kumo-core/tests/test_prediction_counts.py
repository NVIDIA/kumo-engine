# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest
from kumo_core.prediction_counts import (
    MAX_CLASS_HISTOGRAMS,
    QUANTILES,
    count_predictions,
    prediction_summaries,
)
from kumo_core.score_counts import PROBABILITY_EDGES, Counts, merged
from kumo_core.summaries import Histogram, Scalar, Table
from kumo_core.types import TaskType


def _brute_histogram(values, edges):
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[sum(edge <= value for edge in edges)] += 1
    return tuple(counts)


def _batches(task_type, edges, columns, prob=None, pred=None, values=None):
    rows = len(values if prob is None else prob)
    parts = []
    for part in np.array_split(np.arange(rows), 5):
        counts = Counts(dict(edges))
        count_predictions(counts, task_type,
                          prob=None if prob is None else prob[part],
                          pred=None if pred is None else pred[part],
                          target_pred=None if values is None else values[part],
                          columns=columns)
        parts.append(counts)
    return merged(parts)


def _by_name(summaries):
    return {summary.name: summary for summary in summaries}


def _check_column(logged, prefix, values, edges):
    assert logged[f'{prefix}/hist'] == Histogram(
        f'{prefix}/hist', edges, _brute_histogram(values, edges))
    assert logged[f'{prefix}/mean'].value == pytest.approx(values.mean())
    assert logged[f'{prefix}/min'] == Scalar(f'{prefix}/min', values.min())
    assert logged[f'{prefix}/max'] == Scalar(f'{prefix}/max', values.max())
    edges = np.array(edges)
    for q in QUANTILES:
        exact = np.quantile(values, q / 100, method='inverted_cdf')
        low = max([values.min(), *edges[edges <= exact]])
        high = min([values.max(), *edges[edges > exact]])
        assert low <= logged[f'{prefix}/q{q:02d}'].value <= high


def test_binary_counts_match_brute_force():
    rng = np.random.default_rng(0)
    true = rng.random(301)
    prob = np.stack([1 - true, true], axis=1)
    pred = true >= 0.4
    counts = _batches(TaskType.BINARY_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES}, ['False_PROB', 'True_PROB'],
                      prob=prob, pred=pred)

    logged = _by_name(prediction_summaries(counts, threshold=0.4,
                                           baseline=None))
    assert sorted(logged) == sorted([
        f'pred/{column}/{stat}' for column in ('False_PROB', 'True_PROB')
        for stat in ('hist', 'mean', 'min', 'max', *(f'q{q:02d}'
                                                     for q in QUANTILES))
    ] + ['pred/positive_rate'])
    _check_column(logged, 'pred/True_PROB', true, PROBABILITY_EDGES)
    _check_column(logged, 'pred/False_PROB', 1 - true, PROBABILITY_EDGES)
    assert logged['pred/positive_rate'].value == pred.mean()
    assert 'pred/positive_rate' not in _by_name(
        prediction_summaries(counts, threshold=None, baseline=None))


def test_multiclass_summarizes_the_most_predicted_classes():
    rng = np.random.default_rng(1)
    prob = rng.dirichlet(np.arange(1, 26) / 5, size=400)
    columns = [f'c{index}_PROB' for index in range(25)]
    counts = _batches(TaskType.MULTICLASS_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES}, columns, prob=prob)

    logged = _by_name(
        prediction_summaries(counts, threshold=None, baseline=None))
    top = np.bincount(prob.argmax(axis=1), minlength=25)
    order = sorted(range(25), key=lambda index: -top[index])
    assert {name.split('/')[1]
            for name in logged if name.endswith('/hist')
            } == {'confidence', *(columns[index] for index in order[:20])}
    _check_column(logged, 'pred/confidence', prob.max(axis=1),
                  PROBABILITY_EDGES)
    _check_column(logged, f'pred/{columns[order[0]]}', prob[:, order[0]],
                  PROBABILITY_EDGES)
    assert logged['pred/classes'] == Table(
        'pred/classes', ('class', 'rows', 'share'),
        tuple((f'c{index}', int(top[index]), top[index] / 400)
              for index in order))


def test_many_classes_keep_only_their_tally_and_confidence():
    rng = np.random.default_rng(3)
    size = MAX_CLASS_HISTOGRAMS + 1
    prob = rng.dirichlet(np.ones(size), size=300)
    counts = _batches(TaskType.MULTICLASS_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES},
                      [f'c{index}_PROB' for index in range(size)], prob=prob)

    assert set(counts.extents) == {'confidence'}
    logged = _by_name(prediction_summaries(counts, threshold=None,
                                           baseline=None))
    assert [name for name in logged if name.endswith('/hist')] == [
        'pred/confidence/hist']
    top = np.bincount(prob.argmax(axis=1), minlength=size)
    assert logged['pred/classes'].rows[0] == (
        str(top.argmax()), top.max(), top.max() / 300)


def test_regression_counts_on_the_target_bins_and_drift():
    rng = np.random.default_rng(2)
    values = rng.normal(2.0, 1.5, 500)
    edges = (0.0, 1.0, 1.5, 2.0, 3.0, 5.0)
    counts = _batches(TaskType.REGRESSION, {'target': edges}, [], values=values)
    holdout = rng.normal(1.5, 1.0, 300)
    baseline = Counts({'target': edges})
    baseline.histogram('TARGET_PRED', 'target', holdout)

    logged = _by_name(
        prediction_summaries(counts, threshold=None, baseline=baseline))
    _check_column(logged, 'pred', values, edges)
    p = [count / 500 for count in _brute_histogram(values, edges)]
    q = [count / 300 for count in _brute_histogram(holdout, edges)]
    assert logged['drift/score_ks'].value == pytest.approx(
        max(abs(sum(p[:i + 1]) - sum(q[:i + 1])) for i in range(len(p))))
    p, q = ([max(share, 1e-6) for share in shares] for shares in (p, q))
    assert logged['drift/score_psi'].value == pytest.approx(
        sum((a - b) * math.log(a / b) for a, b in zip(p, q)))
    other = Counts({'target': (0.0, 1.0)})
    other.histogram('TARGET_PRED', 'target', holdout)
    assert not any(
        name.startswith('drift/') for name in _by_name(
            prediction_summaries(counts, threshold=None, baseline=other)))


def test_class_names_are_safe_metric_names():
    prob = np.array([[0.7, 0.3], [0.4, 0.6]])
    counts = _batches(TaskType.MULTICLASS_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES}, ['a/b_PROB', 'c@d_PROB'],
                      prob=prob)

    names = {name for name in _by_name(prediction_summaries(
        counts, threshold=None, baseline=None)) if name.endswith('/hist')}
    assert names == {'pred/confidence/hist', 'pred/a_b_PROB/hist',
                     'pred/c_d_PROB/hist'}


@pytest.mark.parametrize('steps', [3, 30])
def test_forecasts_count_each_kept_horizon_on_the_target_bins(steps):
    from kumo_core.evaluation_counts import _horizons

    rng = np.random.default_rng(4)
    values = rng.normal(size=40 * steps)
    horizon = np.tile(np.arange(steps), 40)
    edges = (-1.0, 0.0, 1.0)
    parts = []
    for part in np.array_split(np.arange(40), 4):
        rows = np.concatenate([np.arange(i * steps, (i + 1) * steps)
                               for i in part])
        counts = Counts({'target': edges})
        count_predictions(counts, TaskType.FORECASTING, prob=None, pred=None,
                          target_pred=values[rows], columns=[],
                          horizon=horizon[rows])
        parts.append(counts)

    logged = _by_name(prediction_summaries(merged(parts), threshold=None,
                                           baseline=None))
    kept = _horizons(steps)
    assert len(kept) == min(steps, 12)
    assert [name for name in logged if name.endswith('/hist')] == [
        'pred/hist', *(f'pred/h{h + 1}/hist' for h in kept)]
    for h in kept:
        assert logged[f'pred/h{h + 1}/hist'].counts == _brute_histogram(
            values[horizon == h], edges)
    _check_column(logged, 'pred', values, edges)


def test_a_quantile_at_a_bin_edge_stays_in_the_bin_that_reaches_it():
    values = np.array([0.105] * 99 + [0.505])
    counts = Counts({'score': PROBABILITY_EDGES})
    count_predictions(counts, TaskType.BINARY_CLASSIFICATION,
                      prob=np.stack([1 - values, values], axis=1),
                      pred=values > 0.5, target_pred=None,
                      columns=['False_PROB', 'True_PROB'])

    logged = _by_name(prediction_summaries(counts, threshold=None,
                                           baseline=None))
    _check_column(logged, 'pred/True_PROB', values, PROBABILITY_EDGES)
    assert logged['pred/True_PROB/q99'].value == pytest.approx(0.11)
