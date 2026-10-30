# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from kumo_core.evaluation_counts import (
    baseline,
    evaluation_counts,
    top_score_counts,
)
from kumo_core.prediction_counts import (
    MAX_CLASS_HISTOGRAMS,
    count_predictions,
    count_recommendations,
    count_top_scores,
    item_counts,
    load_counts,
    prediction_summaries,
    save_counts,
)
from kumo_core.score_counts import PROBABILITY_EDGES, Counts, merged
from kumo_core.score_drift import score_drift
from kumo_core.summaries import Histogram, Note, Scalar, Series, Table
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
    return {summary.name: summary for summary in summaries
            if not isinstance(summary, Note)}


def _summary_row(column, values):
    return (column, pytest.approx(values.mean()), values.min(), values.max())


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
    assert sorted(logged) == ['pred/histogram/True_PROB', 'pred/positive_rate',
                              'pred/summary']
    assert logged['pred/histogram/True_PROB'] == Histogram(
        'pred/histogram/True_PROB', PROBABILITY_EDGES,
        _brute_histogram(true, PROBABILITY_EDGES))
    summary = logged['pred/summary']
    assert summary.brief and summary.columns == ('column', 'mean', 'min', 'max')
    assert summary.rows == (_summary_row('True_PROB', true),)
    assert logged['pred/positive_rate'].value == pred.mean()
    assert 'pred/positive_rate' not in _by_name(
        prediction_summaries(counts, threshold=None, baseline=None))


def test_multiclass_summarizes_each_class_and_the_most_predicted():
    rng = np.random.default_rng(1)
    prob = rng.dirichlet(np.arange(1, 26) / 5, size=400)
    columns = [f'c{index}_PROB' for index in range(25)]
    counts = _batches(TaskType.MULTICLASS_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES}, columns, prob=prob)

    logged = _by_name(
        prediction_summaries(counts, threshold=None, baseline=None))
    top = np.bincount(prob.argmax(axis=1), minlength=25)
    order = sorted(range(25), key=lambda index: -top[index])
    assert [name for name in logged if name.startswith('pred/histogram/')] == [
        'pred/histogram/confidence',
        *(f'pred/histogram/{columns[index]}' for index in order[:20])]
    assert logged['pred/histogram/confidence'].counts == _brute_histogram(
        prob.max(axis=1), PROBABILITY_EDGES)
    assert logged['pred/by_class'].columns == (
        'class', 'share predicted', 'mean', 'min', 'max')
    assert logged['pred/by_class'].rows == tuple(
        (f'c{index}', top[index] / 400,
         *_summary_row(None, prob[:, index])[1:]) for index in order)
    assert logged['pred/summary'].rows == (
        _summary_row('confidence', prob.max(axis=1)),)


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
    assert [name for name in logged if name.startswith('pred/histogram/')
            ] == ['pred/histogram/confidence']
    top = np.bincount(prob.argmax(axis=1), minlength=size)
    assert logged['pred/by_class'].rows[0] == (
        str(top.argmax()), top.max() / 300, None, None, None)


def test_regression_counts_on_the_target_bins_and_drift():
    rng = np.random.default_rng(2)
    values = rng.normal(2.0, 1.5, 500)
    edges = (0.0, 1.0, 1.5, 2.0, 3.0, 5.0)
    counts = _batches(TaskType.REGRESSION, {'target': edges}, [], values=values)
    holdout = rng.normal(1.5, 1.0, 300)
    reference = Counts({'target': edges})
    reference.histogram('TARGET_PRED', 'target', holdout)
    reference.extent('TARGET_PRED', holdout)

    summaries = prediction_summaries(counts, threshold=None,
                                     baseline=reference)
    logged = _by_name(summaries)
    assert logged['pred/histogram/predicted'].counts == _brute_histogram(
        values, edges)
    assert logged['pred/summary'].rows == (_summary_row('predicted', values),)
    assert {name: logged[f'drift/{name}'].value
            for name in ('psi', 'ks', 'wasserstein')} == score_drift(
                counts, reference)
    assert summaries[-1] == Note(
        'notes', "Drift compares `predicted` with the holdout over the "
        "training target's 7 bins; PSI groups them at the holdout's deciles.")
    other = Counts({'target': (0.0, 1.0)})
    other.histogram('TARGET_PRED', 'target', holdout)
    assert not any(
        name.startswith('drift/') for name in _by_name(
            prediction_summaries(counts, threshold=None, baseline=other)))


def test_class_names_are_safe_column_names():
    prob = np.array([[0.7, 0.3], [0.4, 0.6]])
    counts = _batches(TaskType.MULTICLASS_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES}, ['a/b_PROB', 'c@d_PROB'],
                      prob=prob)

    names = {name for name in _by_name(prediction_summaries(
        counts, threshold=None, baseline=None))
        if name.startswith('pred/histogram/')}
    assert names == {'pred/histogram/confidence', 'pred/histogram/a_b_PROB',
                     'pred/histogram/c@d_PROB'}


@pytest.mark.parametrize('steps', [3, 30])
def test_forecasts_count_each_kept_horizon_and_drift_from_the_holdout(steps):
    from kumo_core.evaluation_counts import _horizons

    rng = np.random.default_rng(4)
    values = rng.normal(size=40 * steps)
    horizon = np.tile(np.arange(steps), 40)
    holdout = baseline(evaluation_counts(
        TaskType.FORECASTING, pred=values[::-1], target=values,
        target_edges=(-1.0, 0.0, 1.0), horizon=horizon))
    holdout = Counts.from_dict(json.loads(json.dumps(holdout.to_dict())))
    edges = holdout.edges['target']
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
                                           baseline=holdout))
    kept = _horizons(steps)
    assert len(kept) == min(steps, 12)
    assert [name for name in logged if name.startswith('pred/histogram/')] == [
        'pred/histogram/predicted',
        *(f'pred/histogram/predicted, horizon {h + 1}' for h in kept)]
    for h in kept:
        assert logged[f'pred/histogram/predicted, horizon {h + 1}'].counts == \
            _brute_histogram(values[horizon == h], edges)
    assert logged['drift/psi'].value == pytest.approx(0.0)


def test_multilabel_counts_each_labels_predicted_rows_and_top_score():
    rng = np.random.default_rng(5)
    prob = rng.random((300, 25)) * np.linspace(0.2, 1.0, 25)
    pred = prob > 0.5
    columns = [f'l{index}_PROB' for index in range(25)]
    counts = _batches(TaskType.MULTILABEL_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES,
                       'top_score': PROBABILITY_EDGES}, columns, prob=prob,
                      pred=pred)

    logged = _by_name(prediction_summaries(counts, threshold=None,
                                           baseline=None))
    positives = pred.sum(axis=0)
    order = sorted(range(25), key=lambda index: -positives[index])
    assert [name for name in logged if name.startswith('pred/histogram/')] == [
        'pred/histogram/top score',
        *(f'pred/histogram/{columns[index]}' for index in order[:20])]
    assert logged['pred/histogram/top score'].counts == _brute_histogram(
        prob.max(axis=1), PROBABILITY_EDGES)
    assert logged['pred/summary'].rows == (
        _summary_row('top score', prob.max(axis=1)),)
    assert [row[:2] for row in logged['pred/by_class'].rows] == [
        (f'l{index}', positives[index] / 300) for index in order]
    per_row = np.bincount(pred.sum(axis=1))
    assert logged['pred/by_label_count/entities'] == Series(
        'pred/by_label_count/entities',
        tuple((k, float(rows)) for k, rows in enumerate(per_row) if rows),
        'labels')


def test_recommendations_count_items_and_top_scores_across_workers(tmp_path):
    rng = np.random.default_rng(6)
    catalogue = 1000
    items = np.minimum(rng.zipf(1.5, size=(500, 5)) - 1, catalogue)
    items[::7, -1] = -1
    scores = rng.normal(size=500)
    holdout = top_score_counts(rng.normal(size=300))
    parts = []
    for worker, rows in enumerate(np.array_split(np.arange(500), 3)):
        counts = Counts({'top_score': holdout.edges['top_score']})
        for batch in np.array_split(rows, 4):
            count_recommendations(counts, items[batch], catalogue)
            count_top_scores(counts, scores[batch])
        save_counts(counts, tmp_path / f'{worker}.json')
        parts.append(load_counts(tmp_path / f'{worker}.json'))
    assert {path.suffix for path in tmp_path.iterdir()} == {'.json', '.npy'}

    total = item_counts(merged(parts))
    logged = _by_name(prediction_summaries(total, threshold=None,
                                           baseline=holdout))
    valid = items[(items >= 0) & (items < catalogue)]
    per_item = np.bincount(valid, minlength=catalogue)
    bounds = [0, *(2**power for power in range(
        int(np.log2(per_item.max())) + 1)), per_item.max() + 1]
    assert logged['pred/item_frequency'] == Table(
        'pred/item_frequency',
        ('times recommended', 'items', 'share of recommendations'),
        tuple((str(low) if high - 1 <= low else f'{low}–{high - 1}',
               int(np.sum((per_item >= low) & (per_item < high))),
               pytest.approx(per_item[(per_item >= low) & (per_item < high)]
                             .sum() / valid.size))
              for low, high in zip(bounds, bounds[1:])))
    assert logged['pred/item_coverage'].value == np.unique(valid).size / 1000
    assert logged['pred/top_1pct_items_share'].value == pytest.approx(
        np.sort(per_item)[-10:].sum() / valid.size)
    assert logged['pred/histogram/top score'].counts == _brute_histogram(
        scores, holdout.edges['top_score'])
    assert logged['drift/ks'] == Scalar(
        'drift/ks', score_drift(total, holdout)['ks'])


def test_forecast_horizons_are_named_by_their_window():
    counts = Counts({'target': (0.0, 1.0)})
    count_predictions(counts, TaskType.FORECASTING, prob=None, pred=None,
                      target_pred=np.array([0.5, 2.0]), columns=[],
                      horizon=np.array([0, 1]))

    names = [name for name in _by_name(prediction_summaries(
        counts, threshold=None, baseline=None, timeframe=(0, 7, 7, 'days')))
        if name.startswith('pred/histogram/predicted, ')]
    assert names == ['pred/histogram/predicted, days 0–7',
                     'pred/histogram/predicted, days 7–14']
