# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from kumo_core.evaluation_counts import (
    at_k_summaries,
    evaluation_counts,
    evaluation_summaries,
)
from kumo_core.score_counts import PROBABILITY_EDGES, merged
from kumo_core.summaries import (
    Confusion,
    Histogram,
    Note,
    Scalar,
    Series,
    Table,
)
from kumo_core.types import TaskType


def _summaries(task_type, threshold=None, classes=(), timeframe=None,
               **columns):
    halves = [{name: value[part] for name, value in columns.items()
               if not isinstance(value, tuple)}
              for part in np.array_split(np.arange(len(columns['target'])), 2)]
    edges = columns.get('target_edges', ())
    counts = merged([evaluation_counts(task_type, threshold=threshold,
                                       target_edges=edges, **half)
                     for half in halves])
    summaries = evaluation_summaries(task_type, counts, threshold, classes,
                                     timeframe)
    return {summary.name: summary for summary in summaries}


def _bin(value, edges):
    return sum(edge <= value for edge in edges)


def _rows(table):
    return [dict(zip(table.columns, row)) for row in table.rows]


def _approx_rows(table, expected):
    assert len(table.rows) == len(expected)
    for row, wanted in zip(_rows(table), expected):
        assert row == pytest.approx(wanted)


def _scalars(summaries):
    assert all(s.role == 'model' for s in summaries.values()
               if isinstance(s, Scalar))
    return {name: s.value for name, s in summaries.items()
            if isinstance(s, Scalar)}


def test_binary_summaries_match_brute_force():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(1996), [0.0, 0.3, 0.42, 1.0]])
    y = rng.random(p.size) < p
    prob = np.stack([1 - p, p], axis=1)

    summaries = _summaries(TaskType.BINARY_CLASSIFICATION, threshold=0.42,
                           prob=prob, target=y.astype(float))

    rows = _rows(summaries['holdout/by_threshold'])
    assert [row['threshold'] for row in rows] == list(PROBABILITY_EDGES)
    for row in rows:
        hit = p >= row['threshold']
        tp, fp = (hit & y).sum(), (hit & ~y).sum()
        assert row == pytest.approx({
            'threshold': row['threshold'], 'share at or above': hit.mean(),
            'tp': tp, 'fp': fp, 'fn': (~hit & y).sum(),
            'tn': (~hit & ~y).sum(),
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / y.sum(), 'fpr': fp / (~y).sum()})
    bins = np.array([_bin(v, PROBABILITY_EDGES) for v in p])
    bounds = [0.0, *PROBABILITY_EDGES, 1.0]
    _approx_rows(summaries['holdout/calibration'], [
        {'from': bounds[b], 'to': bounds[b + 1], 'rows': (bins == b).sum(),
         'mean predicted': p[bins == b].mean(),
         'mean observed': y[bins == b].mean()}
        for b in sorted(set(bins))])
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    hit = p > 0.42
    tp, fp = (hit & y).sum(), (hit & ~y).sum()
    fn, tn = (~hit & y).sum(), (~hit & ~y).sum()
    assert summaries['holdout/at_threshold_0.42'] == Table(
        'holdout/at_threshold_0.42',
        ('actual', 'predicted positive', 'predicted negative'),
        (('positive', tp, fn), ('negative', fp, tn)), brief=True)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    assert _scalars(summaries) == pytest.approx({
        'logloss': -np.mean(y * np.log(clipped) +
                            (1 - y) * np.log(1 - clipped)),
        'brier': np.mean((p - y)**2),
        'accuracy': (tp + tn) / p.size,
        'precision': precision, 'recall': recall,
        'f1': 2 * precision * recall / (precision + recall),
    })


@pytest.mark.parametrize('threshold, title', [
    (0.5, '0.50'), (0.07, '0.07'), (0.375, '0.375'), (0.123456, '0.1235')])
def test_the_at_threshold_table_is_titled_by_its_threshold(threshold, title):
    p = np.array([0.1, 0.4, 0.6, 0.9])
    summaries = _summaries(TaskType.BINARY_CLASSIFICATION, threshold,
                           prob=p, target=np.array([0, 1, 0, 1]))
    assert f'holdout/at_threshold_{title}' in summaries


@pytest.mark.parametrize('size', [4, 25])
def test_multiclass_summaries_match_brute_force(size):
    rng = np.random.default_rng(1)
    prob = rng.dirichlet(np.ones(size), 1000)
    target = rng.integers(0, size, 1000)
    classes = [f'c{i}' for i in range(size)]

    summaries = _summaries(TaskType.MULTICLASS_CLASSIFICATION, prob=prob,
                           target=target, classes=classes)

    predicted = prob.argmax(1)
    support = np.bincount(target, minlength=size)
    f1 = {}
    for c in range(size):
        tp = ((predicted == c) & (target == c)).sum()
        precision = tp / max((predicted == c).sum(), 1)
        recall = tp / max(support[c], 1)
        f1[c] = 2 * precision * recall / (precision + recall) if tp else 0.0
    top = sorted(range(size), key=lambda c: -support[c])[:20]
    _approx_rows(summaries['holdout/by_class'], [
        {'class': f'c{c}', 'support': support[c],
         'precision': ((predicted == c) & (target == c)).sum() /
         max((predicted == c).sum(), 1),
         'recall': ((predicted == c) & (target == c)).sum() / support[c],
         'f1': f1[c]} for c in top])
    assert _scalars(summaries) == pytest.approx({
        'logloss': -np.mean(np.log(prob[np.arange(target.size), target])),
        'f1_macro': np.mean([f1[c] for c in range(size)
                             if support[c] or (predicted == c).any()]),
        'f1_micro': np.mean(predicted == target),
        'f1_weighted': sum(f1[c] * support[c] for c in range(size)) /
        target.size,
    })
    ranked = np.argsort(-prob, axis=1)
    top_k = summaries['holdout/by_k/accuracy']
    assert top_k.x_label == 'k'
    assert [k for k, _ in top_k.points] == list(range(1, min(5, size) + 1))
    assert [value for _, value in top_k.points] == pytest.approx(
        [(ranked[:, :k] == target[:, None]).any(1).mean()
         for k, _ in top_k.points])
    confusion = summaries['holdout/confusion']
    assert isinstance(confusion, Confusion)
    assert sum(map(sum, confusion.counts)) == target.size
    assert len(confusion.labels) == min(size, 20) + (size > 20)
    confidence = prob.max(1)
    bins = np.array([_bin(v, PROBABILITY_EDGES) for v in confidence])
    assert [(row['rows'], row['accuracy'])
            for row in _rows(summaries['holdout/calibration'])] == \
        pytest.approx([((bins == b).sum(),
                        np.mean(predicted[bins == b] == target[bins == b]))
                       for b in sorted(set(bins))])


def test_regression_summaries_match_brute_force():
    rng = np.random.default_rng(2)
    target = rng.normal(0, 1, 3000)
    pred = target + rng.normal(0.1, 0.5, 3000)
    target[0] = 0.0
    edges = tuple(np.linspace(-3, 3, 13).tolist())

    summaries = _summaries(TaskType.REGRESSION, pred=pred, target=target,
                           target_edges=edges)

    error = pred - target
    nonzero = target != 0
    assert _scalars(summaries) == pytest.approx({
        'mae': np.abs(error).mean(),
        'rmse': np.sqrt((error**2).mean()),
        'r2': 1 - (error**2).sum() / ((target - target.mean())**2).sum(),
        'bias': error.mean(),
        'mape': np.mean(np.abs(error[nonzero]) / np.abs(target[nonzero])),
        'smape': np.mean(2 * np.abs(error) / (np.abs(target) + np.abs(pred))),
    })
    residual = summaries['holdout/residuals/rows']
    assert list(residual.counts) == np.bincount(
        [_bin(v, residual.edges) for v in error],
        minlength=len(residual.edges) + 1).tolist()
    for name, values in (('actual', target), ('predicted', pred)):
        assert summaries[f'holdout/histogram/{name}'] == Histogram(
            f'holdout/histogram/{name}', edges, tuple(np.bincount(
                [_bin(v, edges) for v in values], minlength=14).tolist()),
            (values.min(), values.max()))
    bins = np.array([_bin(v, edges) for v in pred])
    bounds = [pred.min(), *edges, pred.max()]
    _approx_rows(summaries['holdout/calibration'], [
        {'from': bounds[b], 'to': bounds[b + 1], 'rows': (bins == b).sum(),
         'mean predicted': pred[bins == b].mean(),
         'mean actual': target[bins == b].mean()}
        for b in sorted(set(bins))])


def test_multilabel_summaries_match_brute_force():
    rng = np.random.default_rng(3)
    prob = rng.random((800, 6))
    target = rng.random((800, 6)) < prob
    pred = prob > 0.5

    summaries = _summaries(TaskType.MULTILABEL_CLASSIFICATION, prob=prob,
                           pred=pred.astype(int), target=target.astype(float),
                           classes=list('abcdef'))

    tp, fp, fn = ((pred & target).sum(), (pred & ~target).sum(),
                  (~pred & target).sum())
    assert _scalars(summaries) == pytest.approx({
        'f1_micro': 2 * tp / (2 * tp + fp + fn),
        'precision_micro': tp / (tp + fp), 'recall_micro': tp / (tp + fn),
        'f1_macro': np.mean([2 * (pred[:, i] & target[:, i]).sum() / (
            pred[:, i].sum() + target[:, i].sum()) for i in range(6)])})
    by_label = {row['label']: row
                for row in _rows(summaries['holdout/by_label'])}
    for i, label in enumerate('abcdef'):
        assert by_label[label]['precision'] == pytest.approx(
            (pred[:, i] & target[:, i]).sum() / pred[:, i].sum())
    for row in _rows(summaries['holdout/by_threshold']):
        assert row['recall'] == pytest.approx(
            np.mean(prob[target] >= row['threshold']))
    for name, values in (('actual entities', target),
                         ('predicted entities', pred)):
        series = summaries[f'holdout/by_label_count/{name}']
        assert series.x_label == 'labels'
        assert dict(series.points) == {
            n: np.sum(values.sum(1) == n)
            for n in sorted({*target.sum(1), *pred.sum(1)})}


@pytest.mark.parametrize('timeframe, windows', [
    (None, ['horizon 1', 'horizon 2', 'horizon 3']),
    ((0, 7, 7, 'days'), ['(0, 7] days', '(7, 14] days', '(14, 21] days']),
])
def test_forecasting_summaries_are_per_horizon(timeframe, windows):
    rng = np.random.default_rng(4)
    horizon = np.tile(np.arange(3), 500)
    target = rng.normal(size=horizon.size)
    pred = target + rng.normal(0, 0.1 * (horizon + 1))
    edges = tuple(np.linspace(-3, 3, 7).tolist())

    summaries = _summaries(TaskType.FORECASTING, pred=pred, target=target,
                           horizon=horizon, target_edges=edges,
                           timeframe=timeframe)

    rows = _rows(summaries['holdout/by_horizon'])
    for h, row in enumerate(rows):
        kept = horizon == h
        error = pred[kept] - target[kept]
        assert row == pytest.approx({
            'horizon': h + 1, **({} if timeframe is None else
                                 {'window': windows[h]}),
            'rows': kept.sum(),
            'mae': np.abs(error).mean(),
            'rmse': np.sqrt((error**2).mean()),
            'bias (pred − actual)': error.mean(),
            'r2': 1 - (error**2).sum() / (
                (target[kept] - target[kept].mean())**2).sum()})
        assert sum(summaries[f'holdout/histogram/predicted, {windows[h]}']
                   .counts) == kept.sum()
    assert len(rows) == 3
    assert summaries['holdout/calibration'].columns == (
        'from', 'to', 'rows', 'mean predicted', 'mean actual')
    assert summaries['mae'].value == pytest.approx(
        np.abs(pred - target).mean())


def test_ranking_top_scores_are_counted_on_their_percentiles():
    rng = np.random.default_rng(5)
    prob = rng.normal(size=(500, 30))

    counts = merged([evaluation_counts(TaskType.MULTILABEL_RANKING,
                                       prob=prob, target=None)])

    top = prob.max(axis=1)
    edges = tuple(np.unique(np.quantile(top, np.linspace(0, 1, 101))))
    assert counts.edges['top_score'] == edges
    assert counts.arrays['top_score'].tolist() == np.bincount(
        [_bin(v, edges) for v in top], minlength=len(edges) + 1).tolist()
    assert counts.extents['top_score'] == (top.min(), top.max())
    assert evaluation_summaries(TaskType.MULTILABEL_RANKING, counts) == []


def test_ranking_metrics_become_at_k_columns():
    assert at_k_summaries({'test_map@10': 0.2, 'test_map@1': 0.5,
                           'test_loss': 1.0, 'val_map@10': 0.3}) == [
        Series('holdout/by_k/map', ((1, 0.5), (10, 0.2)), 'k')]


def test_at_k_columns_describe_the_refit_model_when_there_is_one():
    assert at_k_summaries({'test_map@10': 0.2, 'test_refit_map@10': 0.25,
                           'test_refit_map@1': 0.6,
                           'test_refit_loss': 1.0}) == [
        Series('holdout/by_k/map', ((1, 0.6), (10, 0.25)), 'k')]


def test_at_k_columns_without_a_holdout_loss_say_so():
    assert at_k_summaries({'test_map@1': 0.5, 'test_acc': 0.9}) == [
        Series('holdout/by_k/map', ((1, 0.5),), 'k'),
        Note('notes', 'Link prediction reports no holdout loss.')]
    assert at_k_summaries({'test_acc': 0.9}) == []
