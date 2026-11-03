# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from functools import reduce

import numpy as np
import pytest
from kumo_core.evaluation_counts import baseline, evaluation_counts
from kumo_core.prediction import COUNTS_DIR, COUNTS_FILE, _merge_counts
from kumo_core.prediction_counts import (
    ITEMS,
    MAX_CLASS_HISTOGRAMS,
    count_predictions,
    count_recommendations,
    count_top_scores,
    item_counts,
    prediction_summaries,
    save_counts,
)
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.score_drift import score_drift
from kumo_core.summaries import (
    BarChart,
    Histogram,
    Note,
    Scalar,
    Series,
    Table,
)
from kumo_core.top_scores import TopScoreSpill, top_score_counts
from kumo_core.types import TaskType

from tests.support.counts import brute_histogram


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
    return reduce(Counts.merge, parts)


def _by_name(summaries):
    return {summary.name: summary for summary in summaries
            if not isinstance(summary, (Note, BarChart))}


def _charts(summaries):
    return {summary.name: summary for summary in summaries
            if isinstance(summary, BarChart)}


def _summary_row(column, values):
    return (column, pytest.approx(values.mean()), values.min(), values.max())


def test_binary_counts_match_brute_force():
    rng = np.random.default_rng(0)
    true = np.append(rng.random(300), 1.0)
    prob = np.stack([1 - true, true], axis=1)
    pred = true >= 0.4
    counts = _batches(TaskType.BINARY_CLASSIFICATION,
                      {'score': PROBABILITY_EDGES}, ['False_PROB', 'True_PROB'],
                      prob=prob, pred=pred)

    logged = _by_name(prediction_summaries(counts, threshold=0.4,
                                           baseline=None))
    assert sorted(logged) == ['pred/histogram/True_PROB', 'pred/positive_rate',
                              'pred/score_histogram', 'pred/summary']
    per_pct = np.bincount(np.minimum(true * 100, 99).astype(int),
                          minlength=100)
    assert logged['pred/score_histogram'] == Series(
        'pred/score_histogram',
        tuple(enumerate(per_pct.astype(float))), 'score_pct', chart=True)
    assert _charts(prediction_summaries(
        counts, threshold=0.4, baseline=None, entity='users',
        as_of='2026-11-01')) == {'pred/score_histogram': BarChart(
            'pred/score_histogram', 'Scores of 301 users, 2026-11-01',
            ('score', 'users'), PROBABILITY_EDGES,
            tuple(per_pct.astype(float)), (),
            ('this run', 'holdout, scaled to the same total'))}
    assert logged['pred/histogram/True_PROB'] == Histogram(
        'pred/histogram/True_PROB', PROBABILITY_EDGES,
        brute_histogram(true, PROBABILITY_EDGES), (true.min(), true.max()))
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
    tested = rng.dirichlet(np.ones(25), size=200)
    holdout = baseline(evaluation_counts(
        TaskType.MULTICLASS_CLASSIFICATION, prob=tested,
        target=rng.integers(0, 25, 200)))

    summaries = prediction_summaries(counts, threshold=None, baseline=holdout)
    logged, charts = _by_name(summaries), _charts(summaries)
    top = np.bincount(prob.argmax(axis=1), minlength=25)
    at_holdout = np.bincount(tested.argmax(axis=1), minlength=25)
    order = sorted(range(25), key=lambda index: -top[index])
    assert [name for name in logged if name.startswith('pred/histogram/')] == [
        'pred/histogram/confidence',
        *(f'pred/histogram/{columns[index]}' for index in order[:20])]
    assert [name for name in logged
            if name.startswith('pred/score_histogram')] == [
        'pred/score_histogram/confidence',
        *(f'pred/score_histogram/c{index}' for index in order[:20])]
    assert list(charts) == [name for name in logged
                            if name.startswith('pred/score_histogram')]
    confidence = charts['pred/score_histogram/confidence']
    assert confidence.title == 'Scores of 400 entities (confidence) vs holdout'
    assert sum(confidence.line) == pytest.approx(400)
    assert charts[f'pred/score_histogram/c{order[0]}'].line == ()
    assert logged['pred/histogram/confidence'].counts == brute_histogram(
        prob.max(axis=1), PROBABILITY_EDGES)
    assert logged['pred/by_class'].columns == (
        'class', 'share now', 'holdout share', 'mean prob', 'min prob',
        'max prob')
    assert logged['pred/by_class'].rows == tuple(
        (f'c{index}', top[index] / 400, at_holdout[index] / 200,
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
        str(top.argmax()), top.max() / 300, None, None, None, None)


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
    assert logged['pred/histogram/predicted'].counts == brute_histogram(
        values, edges)
    assert not any(isinstance(summary, Series) for summary in summaries)
    bins = brute_histogram(values, edges)
    assert _charts(summaries) == {'pred/prediction_histogram': BarChart(
        'pred/prediction_histogram', 'Predictions for 500 entities vs holdout',
        ("predicted value, on the training target's bins", 'entities'),
        (min(values.min(), holdout.min()), *edges,
         max(values.max(), holdout.max())), bins,
        tuple(np.array(brute_histogram(holdout, edges)) * 500 / 300),
        ('this run', 'holdout, scaled to the same total'))}
    assert logged['pred/summary'].rows == (_summary_row('predicted', values),)
    assert {name: logged[f'drift/{name}'].value
            for name in ('psi', 'ks', 'wasserstein')} == score_drift(
                counts, reference)
    assert summaries[-1] == Note(
        'drift', "Score drift compares `predicted` with the model's holdout "
        "scores at training, on the training target's histogram bins. PSI "
        "groups those bins at the holdout's deciles, into up to 10 groups. "
        'PSI rule of thumb: 0 identical, < 0.1 small, 0.1–0.25 moderate, > '
        '0.25 large shift. KS: the largest gap between the two cumulative '
        'distributions, from 0 to 1. Wasserstein: how far the scores moved '
        'on average, in score units.')
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
    from kumo_core.evaluation_counts import kept_horizons

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

    logged = _by_name(prediction_summaries(reduce(Counts.merge, parts),
                                           threshold=None, baseline=holdout))
    kept = kept_horizons(steps)
    assert len(kept) == min(steps, 12)
    assert [name for name in logged if name.startswith('pred/histogram/')] == [
        'pred/histogram/predicted',
        *(f'pred/histogram/predicted, horizon {h + 1}' for h in kept)]
    for h in kept:
        assert logged[f'pred/histogram/predicted, horizon {h + 1}'].counts == \
            brute_histogram(values[horizon == h], edges)
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
    assert [name for name in logged
            if name.startswith('pred/score_histogram')] == [
        'pred/score_histogram/top_score',
        *(f'pred/score_histogram/l{index}' for index in order[:20])]
    assert logged['pred/histogram/top score'].counts == brute_histogram(
        prob.max(axis=1), PROBABILITY_EDGES)
    assert logged['pred/summary'].rows == (
        _summary_row('top score', prob.max(axis=1)),)
    assert [row[:3] for row in logged['pred/by_class'].rows] == [
        (f'l{index}', positives[index] / 300, None) for index in order]
    per_row = np.bincount(pred.sum(axis=1))
    assert logged['pred/by_label_count/entities'] == Series(
        'pred/by_label_count/entities',
        tuple((k, float(rows)) for k, rows in enumerate(per_row) if rows),
        'labels')


def test_recommendations_count_items_and_top_scores_across_workers(
        tmp_path, tmp_path_factory):
    rng = np.random.default_rng(6)
    catalogue = 1000
    items = np.minimum(rng.zipf(1.5, size=(500, 5)) - 1, catalogue)
    items[::7, -1] = -1
    scores = rng.normal(size=500)
    spill = TopScoreSpill(
        str(tmp_path_factory.mktemp('holdout') / 'top.parquet'))
    spill.write(np.arange(300), rng.normal(size=300))
    holdout = top_score_counts(spill.close())
    workers = tmp_path / COUNTS_DIR
    workers.mkdir()
    for worker, rows in enumerate(np.array_split(np.arange(500), 3)):
        counts = Counts({'top_score': holdout.edges['top_score']})
        for batch in np.array_split(rows, 4):
            count_recommendations(counts, items[batch], catalogue)
            count_top_scores(counts, scores[batch])
        assert counts.arrays[ITEMS].dtype == np.uint32
        save_counts(counts, workers / f'run-{worker}.json')
    assert {path.suffix for path in workers.iterdir()} == {'.json', '.npy'}

    _merge_counts(tmp_path, 'run', 3)
    total = Counts.from_dict(json.loads((tmp_path / COUNTS_FILE).read_text()))
    summaries = prediction_summaries(total, threshold=None, baseline=holdout,
                                     as_of='2026-11-01')
    logged = _by_name(summaries)
    valid = items[(items >= 0) & (items < catalogue)]
    per_item = np.bincount(valid, minlength=catalogue)
    bounds = [0, *(2**power for power in range(
        int(np.log2(per_item.max())) + 1)), per_item.max() + 1]
    assert logged['pred/item_frequency'] == Table(
        'pred/item_frequency',
        ('× recommended', 'items', 'share of recs'),
        tuple((str(low) if high - 1 <= low else f'{low}–{high - 1}',
               int(np.sum((per_item >= low) & (per_item < high))),
               pytest.approx(per_item[(per_item >= low) & (per_item < high)]
                             .sum() / valid.size))
              for low, high in zip(bounds, bounds[1:])))
    assert logged['pred/item_coverage'].value == np.unique(valid).size / 1000
    assert logged['pred/top_1pct_items_share'].value == pytest.approx(
        np.sort(per_item)[-10:].sum() / valid.size)
    assert logged['pred/histogram/top score'].counts == brute_histogram(
        scores, holdout.edges['top_score'])
    assert [name for name in logged
            if name.startswith('pred/score_histogram')] == []
    assert logged['pred/item_popularity'] == Series(
        'pred/item_popularity',
        tuple((rank, float(n)) for rank, n in
              enumerate(np.sort(per_item)[::-1][:100], 1) if n),
        'item_rank', chart=True)
    coverage = logged['pred/item_coverage'].value
    top_share = logged['pred/top_1pct_items_share'].value
    assert _charts(summaries) == {'pred/item_popularity': BarChart(
        'pred/item_popularity',
        f'Item popularity, 2026-11-01: {coverage:.0%} of items recommended, '
        f'top 1% get {top_share:.0%} of recommendations',
        ('item rank', 'times recommended'), tuple(map(float, range(101))),
        tuple(np.sort(per_item)[::-1][:100].astype(float)))}
    assert logged['drift/ks'] == Scalar(
        'drift/ks', score_drift(total, holdout)['ks'])
    assert 'so compare PSI with the previous run.' in summaries[-1].text


def test_forecast_horizons_are_named_by_their_window():
    counts = Counts({'target': (0.0, 1.0)})
    count_predictions(counts, TaskType.FORECASTING, prob=None, pred=None,
                      target_pred=np.array([0.5, 2.0]), columns=[],
                      horizon=np.array([0, 1]))

    names = [name for name in _by_name(prediction_summaries(
        counts, threshold=None, baseline=None, timeframe=(0, 7, 7, 'days')))
        if name.startswith('pred/histogram/predicted, ')]
    assert names == ['pred/histogram/predicted, (0, 7] days',
                     'pred/histogram/predicted, (7, 14] days']


def test_item_counts_do_not_depend_on_the_chunk_size(monkeypatch):
    tally = np.random.default_rng(9).zipf(1.3, 5_003).astype(np.uint32) - 1
    whole = item_counts(Counts({}, {ITEMS: tally}))
    monkeypatch.setattr('kumo_core.prediction_counts._CHUNK', 64)
    chunked = item_counts(Counts({}, {ITEMS: tally}))
    assert chunked.edges == whole.edges
    assert chunked.arrays.keys() == whole.arrays.keys()
    for name, array in whole.arrays.items():
        assert chunked.arrays[name].tolist() == array.tolist(), name


def test_the_items_tally_widens_when_counts_could_overflow(monkeypatch):
    monkeypatch.setattr('kumo_core.prediction_counts._UINT32', 3)
    counts = Counts({})
    count_recommendations(counts, [[0, 1], [1, -1], [1, 5]], 4)
    assert counts.arrays[ITEMS].dtype == np.uint32
    count_recommendations(counts, [[1, 2]], 4)
    assert counts.arrays[ITEMS].dtype == np.uint64
    assert counts.arrays[ITEMS].tolist() == [1, 4, 1, 0]
