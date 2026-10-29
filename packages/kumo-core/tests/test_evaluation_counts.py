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
from kumo_core.summaries import Confusion, Curve, Histogram, Scalar, Series
from kumo_core.types import TaskType


def _summaries(task_type, threshold=None, classes=(), **columns):
    halves = [{name: value[part] for name, value in columns.items()
               if not isinstance(value, tuple)}
              for part in np.array_split(np.arange(len(columns['target'])), 2)]
    edges = columns.get('target_edges', ())
    counts = merged([evaluation_counts(task_type, threshold=threshold,
                                       target_edges=edges, **half)
                     for half in halves])
    summaries = evaluation_summaries(task_type, counts, threshold, classes)
    return {summary.name: summary for summary in summaries}


def _bin(value, edges):
    return sum(edge <= value for edge in edges)


def _points(curve, axis):
    return dict(getattr(curve, axis).points)


def test_binary_summaries_match_brute_force():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(1996), [0.0, 0.3, 0.42, 1.0]])
    y = rng.random(p.size) < p
    prob = np.stack([1 - p, p], axis=1)

    summaries = _summaries(TaskType.BINARY_CLASSIFICATION, threshold=0.42,
                           prob=prob, target=y.astype(float))

    roc, pr = summaries['test/roc'], summaries['test/pr']
    for step, t in enumerate(PROBABILITY_EDGES):
        assert _points(roc, 'y')[step] == pytest.approx(np.mean(p[y] >= t))
        assert _points(roc, 'x')[step] == pytest.approx(np.mean(p[~y] >= t))
        if (p >= t).any():
            assert _points(pr, 'y')[step] == pytest.approx(
                np.mean(y[p >= t]))
    calibration = summaries['test/calibration']
    bins = np.array([_bin(v, PROBABILITY_EDGES) for v in p])
    for bin, observed in _points(calibration, 'y').items():
        assert observed == pytest.approx(y[bins == bin].mean())
        assert _points(calibration, 'x')[bin] == pytest.approx(
            p[bins == bin].mean())
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    hit = p > 0.42
    tp, fp = (hit & y).sum(), (hit & ~y).sum()
    fn, tn = (~hit & y).sum(), (~hit & ~y).sum()
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    assert {name: s.value for name, s in summaries.items()
            if isinstance(s, Scalar)} == pytest.approx({
                'test/logloss': -np.mean(y * np.log(clipped) +
                                         (1 - y) * np.log(1 - clipped)),
                'test/brier': np.mean((p - y)**2),
                'test/label/positive_rate': y.mean(),
                'test/confusion/tp': tp, 'test/confusion/fp': fp,
                'test/confusion/fn': fn, 'test/confusion/tn': tn,
                'test/accuracy': (tp + tn) / p.size,
                'test/precision': precision, 'test/recall': recall,
                'test/f1': 2 * precision * recall / (precision + recall),
            })


@pytest.mark.parametrize('size', [4, 25])
def test_multiclass_summaries_match_brute_force(size):
    rng = np.random.default_rng(1)
    prob = rng.dirichlet(np.ones(size), 1000)
    target = rng.integers(0, size, 1000)
    classes = [f'c{i}' for i in range(size)]

    summaries = _summaries(TaskType.MULTICLASS_CLASSIFICATION, prob=prob,
                           target=target, classes=classes)

    predicted = prob.argmax(1)
    f1 = {}
    for c in range(size):
        tp = ((predicted == c) & (target == c)).sum()
        precision = tp / max((predicted == c).sum(), 1)
        recall = tp / max((target == c).sum(), 1)
        f1[c] = 2 * precision * recall / (precision + recall) if tp else 0.0
        if f'test/class/c{c}/f1' in summaries:
            assert summaries[f'test/class/c{c}/precision'].value == \
                pytest.approx(precision)
            assert summaries[f'test/class/c{c}/f1'].value == pytest.approx(
                f1[c])
    support = np.bincount(target, minlength=size)
    assert summaries['test/accuracy'].value == np.mean(predicted == target)
    assert summaries['test/f1_macro'].value == pytest.approx(
        np.mean([f1[c] for c in range(size)
                 if support[c] or (predicted == c).any()]))
    assert summaries['test/f1_weighted'].value == pytest.approx(
        sum(f1[c] * support[c] for c in range(size)) / target.size)
    assert summaries['test/logloss'].value == pytest.approx(
        -np.mean(np.log(prob[np.arange(target.size), target])))
    ranked = np.argsort(-prob, axis=1)
    top_k = summaries['test/top_k_accuracy'].points
    assert [k for k, _ in top_k] == list(range(1, min(5, size) + 1))
    assert [value for _, value in top_k] == pytest.approx(
        [(ranked[:, :k] == target[:, None]).any(1).mean() for k, _ in top_k])
    confusion = summaries['test/confusion']
    assert isinstance(confusion, Confusion)
    assert sum(map(sum, confusion.counts)) == target.size
    assert len(confusion.labels) == min(size, 20) + (size > 20)
    if size > 20:
        assert confusion.labels[-1] == 'other'
    actual = classes.index(confusion.labels[0])
    assert confusion.counts[0][0] == ((target == actual) &
                                      (predicted == actual)).sum()
    reliability = summaries['test/reliability']
    confidence = prob.max(1)
    bins = np.array([_bin(v, PROBABILITY_EDGES) for v in confidence])
    for bin, accuracy in _points(reliability, 'y').items():
        assert accuracy == pytest.approx(
            np.mean(predicted[bins == bin] == target[bins == bin]))


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
    assert {name: s.value for name, s in summaries.items()
            if isinstance(s, Scalar)} == pytest.approx({
                'test/mae': np.abs(error).mean(),
                'test/rmse': np.sqrt((error**2).mean()),
                'test/r2': 1 - (error**2).sum() / (
                    (target - target.mean())**2).sum(),
                'test/bias': error.mean(),
                'test/mape': np.mean(np.abs(error[nonzero]) /
                                     np.abs(target[nonzero])),
                'test/smape': np.mean(2 * np.abs(error) /
                                      (np.abs(target) + np.abs(pred))),
            })
    residual = summaries['test/residual/hist']
    assert isinstance(residual, Histogram)
    assert list(residual.counts) == np.bincount(
        [_bin(v, residual.edges) for v in error],
        minlength=len(residual.edges) + 1).tolist()
    assert summaries['test/TARGET_PRED/hist'].edges == edges
    calibration = summaries['test/calibration']
    assert isinstance(calibration, Curve)
    bins = np.array([_bin(v, edges) for v in pred])
    for bin, actual in _points(calibration, 'y').items():
        assert actual == pytest.approx(target[bins == bin].mean())


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
    assert summaries['test/f1_micro'].value == pytest.approx(
        2 * tp / (2 * tp + fp + fn))
    for i, label in enumerate('abcdef'):
        hit = (pred[:, i] & target[:, i]).sum()
        assert summaries[f'test/label/{label}/precision'].value == \
            pytest.approx(hit / pred[:, i].sum())
    roc = summaries['test/roc']
    for step, t in enumerate(PROBABILITY_EDGES):
        assert dict(roc.y.points)[step] == pytest.approx(
            np.mean(prob[target] >= t))
    assert dict(summaries['test/labels_per_entity'].points) == {
        n: count for n, count in enumerate(np.bincount(target.sum(1)))
        if count}


def test_forecasting_summaries_are_per_horizon():
    rng = np.random.default_rng(4)
    horizon = np.tile(np.arange(3), 500)
    target = rng.normal(size=horizon.size)
    pred = target + rng.normal(0, 0.1 * (horizon + 1))
    edges = tuple(np.linspace(-3, 3, 7).tolist())

    summaries = _summaries(TaskType.FORECASTING, pred=pred, target=target,
                           horizon=horizon, target_edges=edges)

    for h in range(3):
        rows = horizon == h
        assert dict(summaries['test/horizon/mae'].points)[h + 1] == \
            pytest.approx(np.abs(pred[rows] - target[rows]).mean())
        assert sum(summaries[f'test/TARGET_PRED/h{h + 1}/hist'].counts) == \
            rows.sum()
    assert summaries['test/mae'].value == pytest.approx(
        np.abs(pred - target).mean())


def test_ranking_metrics_become_at_k_series():
    assert at_k_summaries({'test_map@10': 0.2, 'test_map@1': 0.5,
                           'test_loss': 1.0, 'val_map@10': 0.3}) == [
        Series('test/map_at_k', ((1, 0.5), (10, 0.2)), 'k')]
