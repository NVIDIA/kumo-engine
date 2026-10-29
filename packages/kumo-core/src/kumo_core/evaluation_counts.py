# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact test-split counts per task type (`evaluation_counts`), and the
``test/`` summaries derived from them (`evaluation_summaries`).

The counts take the test loop's calibrated scores of every evaluated row.
Binary, multiclass and multilabel scores fall in probability hundredths; a
regression or forecasting target in the training split's bins, a forecast
also per horizon. `baseline` keeps the counts a prediction run's scores are
compared with; `at_k_summaries` turns ranking metrics into ``@k`` series.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.summaries import Confusion, Curve, Scalar, Series, Summary
from kumo_core.types import TaskType

_EPS = 1e-15
_TOP_K = 5
_TOP_CLASSES = 20
_RESIDUAL_BINS = 40
_HORIZONS = 12
#: The arrays a prediction run's score distribution is compared with.
BASELINE = ('True_PROB', 'False_PROB', 'confidence', 'classes', 'TARGET_PRED')


def evaluation_counts(task_type: TaskType, *, target: Any,
                      prob: Any = None, pred: Any = None,
                      threshold: Optional[float] = None,
                      target_edges: Sequence[float] = (),
                      horizon: Any = None) -> Optional[Counts]:
    r"""The counts of one test split; ``None`` for task types without
    them. ``threshold`` labels a binary prediction positive when its
    ``True_PROB`` is above it, as ``TARGET_PRED`` does; ``horizon`` is each
    forecast row's step ahead, from 0."""
    if task_type is TaskType.BINARY_CLASSIFICATION:
        prob = np.asarray(prob, dtype=np.float64)
        return _binary(prob[:, -1] if prob.ndim == 2 else prob,
                       np.asarray(target).reshape(-1) == 1, threshold)
    if task_type is TaskType.MULTICLASS_CLASSIFICATION:
        return _multiclass(np.asarray(prob, dtype=np.float64),
                           np.asarray(target, dtype=np.int64).reshape(-1))
    if task_type is TaskType.MULTILABEL_CLASSIFICATION:
        return _multilabel(np.asarray(prob, dtype=np.float64),
                           np.asarray(pred).astype(bool),
                           np.asarray(target).astype(bool))
    if task_type in (TaskType.REGRESSION, TaskType.FORECASTING):
        values = (np.asarray(pred, dtype=np.float64).reshape(-1),
                  np.asarray(target, dtype=np.float64).reshape(-1),
                  tuple(target_edges))
        return (_regression(*values) if horizon is None else
                _forecasting(*values, np.asarray(horizon, dtype=np.int64)))
    return None


def baseline(counts: Counts) -> Counts:
    r"""``counts`` without what only the test metrics need."""
    return Counts(dict(counts.edges), {
        name: array for name, array in counts.arrays.items()
        if name in BASELINE
    }, dict(counts.extents))


def evaluation_summaries(task_type: TaskType, counts: Counts,
                         threshold: Optional[float] = None,
                         classes: Sequence[Any] = ()) -> List[Summary]:
    if task_type is TaskType.BINARY_CLASSIFICATION:
        return _binary_summaries(counts, threshold)
    if task_type is TaskType.MULTICLASS_CLASSIFICATION:
        return _multiclass_summaries(counts, classes)
    if task_type is TaskType.MULTILABEL_CLASSIFICATION:
        return _multilabel_summaries(counts, classes)
    if task_type is TaskType.REGRESSION:
        return _regression_summaries(counts)
    if task_type is TaskType.FORECASTING:
        return [*_regression_summaries(counts), *_horizon_summaries(counts)]
    return []


def at_k_summaries(metrics: Mapping[str, float]) -> List[Summary]:
    r"""``test_map@1``, ``test_map@10``... as the series ``test/map_at_k``
    with step k."""
    series: Dict[str, List[Tuple[int, float]]] = {}
    for name, value in metrics.items():
        match = re.fullmatch(r'test_(\w+)@(\d+)', name)
        if match:
            series.setdefault(match.group(1), []).append(
                (int(match.group(2)), float(value)))
    return [Series(f'test/{metric}_at_k', tuple(sorted(points)), 'k')
            for metric, points in sorted(series.items())]


def _binary(p: np.ndarray, y: np.ndarray,
            threshold: Optional[float]) -> Counts:
    counts = Counts({'score': PROBABILITY_EDGES})
    counts.histogram('True_PROB', 'score', p)
    counts.histogram('False_PROB', 'score', 1 - p)
    for label, rows in (('positive', y), ('negative', ~y)):
        counts.histogram(label, 'score', p[rows])
        counts.histogram(f'{label}_sum', 'score', p[rows], weights=p[rows])
    clipped = np.clip(p, _EPS, 1 - _EPS)
    counts.total('logloss',
                 -np.where(y, np.log(clipped), np.log1p(-clipped)).sum())
    counts.total('brier', ((p - y)**2).sum())
    counts.total('rows', p.size)
    if threshold is not None:
        hit = p > threshold
        for name, rows in (('tp', hit & y), ('fp', hit & ~y),
                           ('fn', ~hit & y), ('tn', ~hit & ~y)):
            counts.total(name, rows.sum())
    return counts


def _binary_summaries(counts: Counts,
                      threshold: Optional[float]) -> List[Summary]:
    arrays = counts.arrays
    rows = float(arrays['rows'])
    # Predicted positive at the threshold step/100: every bin above it.
    tp = _above(arrays['positive'])
    fp = _above(arrays['negative'])
    positives = max(float(arrays['positive'].sum()), 1.0)
    negatives = max(float(arrays['negative'].sum()), 1.0)
    steps = range(len(PROBABILITY_EDGES))
    flagged = [step for step in steps if tp[step] + fp[step]]
    filled = [bin for bin in range(1, len(PROBABILITY_EDGES))
              if arrays['positive'][bin] + arrays['negative'][bin]]
    summaries: List[Summary] = [
        Scalar('test/logloss', float(arrays['logloss']) / rows),
        Scalar('test/brier', float(arrays['brier']) / rows),
        Scalar('test/label/positive_rate',
               float(arrays['positive'].sum()) / rows),
        _curve('test/roc', 'fpr', 'tpr', steps, fp / negatives,
               tp / positives),
        _curve('test/pr', 'recall', 'precision', flagged, tp / positives,
               tp / np.maximum(tp + fp, 1)),
        _curve('test/gain', 'ppr', 'tpr', steps, (tp + fp) / rows,
               tp / positives),
        _curve('test/calibration', 'predicted', 'observed', filled,
               (arrays['positive_sum'] + arrays['negative_sum']) /
               np.maximum(arrays['positive'] + arrays['negative'], 1),
               arrays['positive'] /
               np.maximum(arrays['positive'] + arrays['negative'], 1),
               x_label='score bin (hundredths)'),
    ]
    if threshold is not None:
        confusion = {name: float(arrays[name])
                     for name in ('tp', 'fp', 'fn', 'tn')}
        precision = _ratio(confusion['tp'], confusion['tp'] + confusion['fp'])
        recall = _ratio(confusion['tp'], confusion['tp'] + confusion['fn'])
        summaries += [
            *(Scalar(f'test/confusion/{name}', value)
              for name, value in confusion.items()),
            Scalar('test/accuracy', (confusion['tp'] + confusion['tn']) / rows),
            Scalar('test/precision', precision),
            Scalar('test/recall', recall),
            Scalar('test/f1', _ratio(2 * precision * recall,
                                     precision + recall)),
        ]
    return summaries


def _multiclass(prob: np.ndarray, target: np.ndarray) -> Counts:
    size = prob.shape[1]
    predicted = prob.argmax(axis=1)
    confidence = prob.max(axis=1)
    counts = Counts({'score': PROBABILITY_EDGES})
    counts.tally('classes', predicted, size)
    counts.tally('confusion', target * size + predicted, size * size)
    counts.histogram('confidence', 'score', confidence)
    counts.histogram('confidence_sum', 'score', confidence,
                     weights=confidence)
    counts.histogram('correct', 'score', confidence,
                     weights=predicted == target)
    ranked = np.argsort(-prob, axis=1, kind='stable')[:, :_TOP_K]
    for k in range(1, min(_TOP_K, size) + 1):
        counts.total(f'top_{k}', (ranked[:, :k] == target[:, None]).any(1)
                     .sum())
    counts.total('logloss', -np.log(np.clip(
        prob[np.arange(target.size), target], _EPS, 1)).sum())
    counts.total('rows', target.size)
    return counts


def _multiclass_summaries(counts: Counts,
                          classes: Sequence[Any]) -> List[Summary]:
    arrays = counts.arrays
    size = len(arrays['classes'])
    names = [str(name) for name in classes] or [str(i) for i in range(size)]
    confusion = arrays['confusion'].reshape(size, size)
    rows = float(arrays['rows'])
    support, predicted, hits = (confusion.sum(axis=1), confusion.sum(axis=0),
                                np.diag(confusion))
    precision = hits / np.maximum(predicted, 1)
    recall = hits / np.maximum(support, 1)
    f1 = np.where(precision + recall > 0,
                  2 * precision * recall / np.maximum(precision + recall,
                                                      _EPS), 0.0)
    seen = (support + predicted) > 0
    top = [int(i) for i in np.argsort(-support, kind='stable')[:_TOP_CLASSES]]
    filled = [bin for bin in range(1, len(PROBABILITY_EDGES))
              if arrays['confidence'][bin]]
    return [
        Scalar('test/accuracy', hits.sum() / rows),
        Scalar('test/logloss', float(arrays['logloss']) / rows),
        Scalar('test/f1_macro', float(f1[seen].mean())),
        Scalar('test/f1_micro', hits.sum() / rows),
        Scalar('test/f1_weighted', float((f1 * support).sum() / rows)),
        Series('test/top_k_accuracy',
               tuple((k, float(arrays[f'top_{k}']) / rows)
                     for k in range(1, min(_TOP_K, size) + 1)),
               x_label='k'),
        *(Scalar(f'test/class/{_metric_part(names[i])}/{name}',
                 float(values[i]))
          for i in top
          for name, values in (('precision', precision), ('recall', recall),
                               ('f1', f1))),
        _confusion(confusion, names, top),
        _curve('test/reliability', 'confidence', 'accuracy', filled,
               arrays['confidence_sum'] / np.maximum(arrays['confidence'], 1),
               arrays['correct'] / np.maximum(arrays['confidence'], 1),
               x_label='confidence bin (hundredths)'),
    ]


def _regression(pred: np.ndarray, target: np.ndarray,
                edges: Tuple[float, ...]) -> Counts:
    error = pred - target
    half = (edges[-1] - edges[0]) / 2 if len(edges) > 1 else 1.0
    counts = Counts({
        'target': edges,
        'residual': tuple(np.linspace(-half or -1.0, half or 1.0,
                                      _RESIDUAL_BINS + 1).tolist()),
    })
    counts.histogram('TARGET_PRED', 'target', pred)
    counts.histogram('TARGET', 'target', target)
    counts.histogram('pred_sum', 'target', pred, weights=pred)
    counts.histogram('target_sum', 'target', pred, weights=target)
    counts.histogram('residual', 'residual', error)
    counts.extent('TARGET_PRED', pred)
    nonzero = target != 0
    scale = np.abs(target) + np.abs(pred)
    for name, value in (
        ('rows', target.size), ('abs_error', np.abs(error).sum()),
        ('squared_error', (error**2).sum()), ('error', error.sum()),
        ('target', target.sum()), ('target_squared', (target**2).sum()),
        ('ape', (np.abs(error[nonzero]) / np.abs(target[nonzero])).sum()),
        ('ape_rows', nonzero.sum()),
        ('sape', (2 * np.abs(error[scale > 0]) / scale[scale > 0]).sum()),
        ('sape_rows', (scale > 0).sum()),
    ):
        counts.total(name, value)
    return counts


def _regression_summaries(counts: Counts) -> List[Summary]:
    arrays = counts.arrays
    rows = float(arrays['rows'])
    total = float(arrays['target'])
    spread = float(arrays['target_squared']) - total * total / rows
    squared = float(arrays['squared_error'])
    filled = [bin for bin, count in enumerate(arrays['TARGET_PRED']) if count]
    return [
        Scalar('test/mae', float(arrays['abs_error']) / rows),
        Scalar('test/rmse', float(np.sqrt(squared / rows))),
        Scalar('test/r2', 1 - squared / spread if spread > 0 else 0.0),
        Scalar('test/bias', float(arrays['error']) / rows),
        Scalar('test/mape', _ratio(float(arrays['ape']),
                                   float(arrays['ape_rows']))),
        Scalar('test/smape', _ratio(float(arrays['sape']),
                                    float(arrays['sape_rows']))),
        counts.summary('residual', 'residual', 'test/residual/hist'),
        counts.summary('TARGET_PRED', 'target', 'test/TARGET_PRED/hist'),
        counts.summary('TARGET', 'target', 'test/TARGET/hist'),
        _curve('test/calibration', 'predicted', 'actual', filled,
               arrays['pred_sum'] / np.maximum(arrays['TARGET_PRED'], 1),
               arrays['target_sum'] / np.maximum(arrays['TARGET_PRED'], 1),
               x_label='prediction bin'),
    ]


def _multilabel(prob: np.ndarray, pred: np.ndarray,
                target: np.ndarray) -> Counts:
    size = prob.shape[1]
    counts = Counts({'score': PROBABILITY_EDGES})
    for name, rows in (('tp', pred & target), ('fp', pred & ~target),
                       ('fn', ~pred & target), ('support', target)):
        counts.tally(name, np.nonzero(rows)[1], size)
    counts.tally('labels', target.sum(axis=1), size + 1)
    counts.tally('predicted_labels', pred.sum(axis=1), size + 1)
    counts.histogram('positive', 'score', prob[target])
    counts.histogram('negative', 'score', prob[~target])
    counts.total('rows', target.shape[0])
    return counts


def _multilabel_summaries(counts: Counts,
                          classes: Sequence[Any]) -> List[Summary]:
    arrays = counts.arrays
    tp, fp, fn = (arrays[name].astype(np.float64) for name in ('tp', 'fp', 'fn'))
    names = [str(name) for name in classes] or [str(i) for i in range(tp.size)]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    seen = (tp + fp + fn) > 0
    top = np.argsort(-arrays['support'], kind='stable')[:_TOP_CLASSES]
    tp_all, fp_all, fn_all = tp.sum(), fp.sum(), fn.sum()
    positives = _above(arrays['positive'])
    negatives = _above(arrays['negative'])
    return [
        Scalar('test/f1_micro', _ratio(2 * tp_all, 2 * tp_all + fp_all + fn_all)),
        Scalar('test/precision_micro', _ratio(tp_all, tp_all + fp_all)),
        Scalar('test/recall_micro', _ratio(tp_all, tp_all + fn_all)),
        Scalar('test/f1_macro', float(f1[seen].mean()) if seen.any() else 0.0),
        *(Scalar(f'test/label/{_metric_part(names[i])}/{name}',
                 float(values[i]))
          for i in top
          for name, values in (('precision', precision), ('recall', recall),
                               ('f1', f1))),
        _curve('test/roc', 'fpr', 'tpr', range(len(PROBABILITY_EDGES)),
               negatives / max(float(arrays['negative'].sum()), 1.0),
               positives / max(float(arrays['positive'].sum()), 1.0)),
        *(Series(f'test/{name}_per_entity',
                 tuple((n, float(count))
                       for n, count in enumerate(arrays[name]) if count),
                 'labels')
          for name in ('labels', 'predicted_labels')),
    ]


def _forecasting(pred: np.ndarray, target: np.ndarray,
                 edges: Tuple[float, ...], horizon: np.ndarray) -> Counts:
    counts = _regression(pred, target, edges)
    size = int(horizon.max()) + 1 if horizon.size else 1
    counts.edges['horizon'] = tuple(float(h) for h in range(1, size))
    error = pred - target
    counts.histogram('horizon_rows', 'horizon', horizon)
    for name, values in (('horizon_abs_error', np.abs(error)),
                         ('horizon_squared_error', error**2),
                         ('horizon_error', error),
                         ('horizon_target', target),
                         ('horizon_target_squared', target**2)):
        counts.histogram(name, 'horizon', horizon, weights=values)
    for h in _horizons(size):
        counts.histogram(f'TARGET_PRED_h{h + 1}', 'target', pred[horizon == h])
    return counts


def _horizon_summaries(counts: Counts) -> List[Summary]:
    arrays = counts.arrays
    rows = np.maximum(arrays['horizon_rows'], 1)
    spread = (arrays['horizon_target_squared'] -
              arrays['horizon_target']**2 / rows)
    r2 = np.where(spread > 0,
                  1 - arrays['horizon_squared_error'] / np.maximum(spread,
                                                                   _EPS), 0.0)
    steps = [h for h, count in enumerate(arrays['horizon_rows']) if count]
    return [
        *(Series(f'test/horizon/{name}',
                 tuple((h + 1, float(values[h])) for h in steps), 'horizon')
          for name, values in (
              ('mae', arrays['horizon_abs_error'] / rows),
              ('rmse', np.sqrt(arrays['horizon_squared_error'] / rows)),
              ('bias', arrays['horizon_error'] / rows),
              ('r2', r2))),
        *(counts.summary(name, 'target',
                         f"test/TARGET_PRED/{name.rsplit('_', 1)[1]}/hist")
          for name in sorted(arrays) if name.startswith('TARGET_PRED_h')),
    ]


def _horizons(size: int) -> List[int]:
    if size <= _HORIZONS:
        return list(range(size))
    return sorted({int(h) for h in np.linspace(0, size - 1, _HORIZONS)
                   .round()})


def _above(counts: np.ndarray) -> np.ndarray:
    r"""``counts`` at or above each edge: index ``k`` sums the bins from
    ``k + 1`` on."""
    return np.cumsum(counts[::-1])[::-1][1:].astype(np.float64)


def _curve(name: str, x: str, y: str, steps: Sequence[int], xs: np.ndarray,
           ys: np.ndarray,
           x_label: str = 'threshold (hundredths)') -> Curve:
    return Curve(name,
                 Series(f'{name}/{x}', tuple((step, float(xs[step]))
                                             for step in steps), x_label),
                 Series(f'{name}/{y}', tuple((step, float(ys[step]))
                                             for step in steps), x_label))


def _confusion(confusion: np.ndarray, names: Sequence[str],
               top: Sequence[int]) -> Confusion:
    rest = [i for i in range(len(names)) if i not in set(top)]
    groups: List[Tuple[str, List[int]]] = [(names[i], [i]) for i in top]
    if rest:
        groups.append(('other', rest))
    return Confusion('test/confusion', tuple(name for name, _ in groups),
                     tuple(tuple(int(confusion[np.ix_(actual, predicted)].sum())
                                 for _, predicted in groups)
                           for _, actual in groups))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _metric_part(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.\- ]', '_', name)


def class_names(target_encoder: Any) -> List[Any]:
    mapping = getattr(target_encoder.tensor_mapper, 'mapping', None)
    return [] if mapping is None else list(mapping)
