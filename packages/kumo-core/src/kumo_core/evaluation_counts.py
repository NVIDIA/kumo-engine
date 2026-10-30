# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact test-split counts per task type (`evaluation_counts`), and the
``holdout/`` summaries derived from them (`evaluation_summaries`).

The counts take the test loop's calibrated scores of every evaluated row.
Binary, multiclass and multilabel scores fall in probability hundredths; a
regression or forecasting target in the training split's bins, a forecast
also per horizon; a ranking's top scores in their own exact percentiles.
`baseline` keeps the counts a prediction run's scores are compared with;
`at_k_summaries` turns ranking metrics into ``@k`` series.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from kumo_core.score_counts import PROBABILITY_EDGES, Counts, bin_bounds
from kumo_core.summaries import (
    Confusion,
    Scalar,
    Series,
    Summary,
    Table,
)
from kumo_core.types import TaskType

_EPS = 1e-15
_TOP_K = 5
_TOP_CLASSES = 20
_RESIDUAL_BINS = 40
_HORIZONS = 12
_TOP_SCORE_BINS = 100
#: The arrays a prediction run's score distribution is compared with.
BASELINE = ('True_PROB', 'confidence', 'top_score', 'classes', 'TARGET_PRED')
#: A forecast's target window: its first horizon's ``(start, end)`` offsets,
#: the ``step`` between horizons and their time ``unit``.
Timeframe = Tuple[int, int, int, str]


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
    if task_type is TaskType.MULTILABEL_RANKING:
        return top_score_counts(np.asarray(prob, dtype=np.float64).max(axis=1))
    if task_type in (TaskType.REGRESSION, TaskType.FORECASTING):
        values = (np.asarray(pred, dtype=np.float64).reshape(-1),
                  np.asarray(target, dtype=np.float64).reshape(-1),
                  tuple(target_edges))
        return (_regression(*values) if horizon is None else
                _forecasting(*values, np.asarray(horizon, dtype=np.int64)))
    return None


def top_score_counts(scores: Any) -> Counts:
    r"""Counts of each entity's top ``scores``, which have no fixed range,
    on the ``top_score`` axis cut at their own exact percentiles."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    edges = () if not scores.size else tuple(np.unique(np.quantile(
        scores, np.linspace(0, 1, _TOP_SCORE_BINS + 1))).tolist())
    counts = Counts({'top_score': edges})
    counts.histogram('top_score', 'top_score', scores)
    counts.extent('top_score', scores)
    return counts


def baseline(counts: Counts) -> Counts:
    r"""``counts`` without what only the test metrics need."""
    return Counts(dict(counts.edges), {
        name: array for name, array in counts.arrays.items()
        if name in BASELINE
    }, dict(counts.extents))


def evaluation_summaries(task_type: TaskType, counts: Counts,
                         threshold: Optional[float] = None,
                         classes: Sequence[Any] = (),
                         timeframe: Optional[Timeframe] = None
                         ) -> List[Summary]:
    r"""The holdout's summaries from its ``counts``; ``timeframe`` names a
    forecast's horizons by their window (``days 0–7``), else by number."""
    if task_type is TaskType.BINARY_CLASSIFICATION:
        return _binary_summaries(counts, threshold)
    if task_type is TaskType.MULTICLASS_CLASSIFICATION:
        return _multiclass_summaries(counts, classes)
    if task_type is TaskType.MULTILABEL_CLASSIFICATION:
        return _multilabel_summaries(counts, classes)
    if task_type is TaskType.REGRESSION:
        return _regression_summaries(counts)
    if task_type is TaskType.FORECASTING:
        return [*_regression_summaries(counts),
                *_horizon_summaries(counts, timeframe)]
    return []


def at_k_summaries(metrics: Mapping[str, float]) -> List[Summary]:
    r"""``test_map@1``, ``test_map@10``... as the column ``map`` of the
    table ``holdout/by_k``."""
    series: Dict[str, List[Tuple[int, float]]] = {}
    for name, value in metrics.items():
        match = re.fullmatch(r'test_(\w+)@(\d+)', name)
        if match:
            series.setdefault(match.group(1), []).append(
                (int(match.group(2)), float(value)))
    return [Series(f'holdout/by_k/{metric}', tuple(sorted(points)), 'k')
            for metric, points in sorted(series.items())]


def _binary(p: np.ndarray, y: np.ndarray,
            threshold: Optional[float]) -> Counts:
    counts = Counts({'score': PROBABILITY_EDGES})
    counts.histogram('True_PROB', 'score', p)
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
    labelled = arrays['positive'] + arrays['negative']
    summaries: List[Summary] = [
        _model('logloss', float(arrays['logloss']) / rows),
        _model('brier', float(arrays['brier']) / rows),
        _by_threshold(arrays['positive'], arrays['negative']),
        _calibration(PROBABILITY_EDGES, (0.0, 1.0), labelled,
                     arrays['positive_sum'] + arrays['negative_sum'],
                     arrays['positive'], ('mean predicted', 'mean observed')),
    ]
    if threshold is not None:
        tp, fp, fn, tn = (int(arrays[name]) for name in ('tp', 'fp', 'fn',
                                                         'tn'))
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        summaries += [
            Table('holdout/at_threshold', ('threshold', 'tp', 'fp', 'fn', 'tn'),
                  ((threshold, tp, fp, fn, tn),), brief=True),
            _model('accuracy', (tp + tn) / rows),
            _model('precision', precision),
            _model('recall', recall),
            _model('f1', _ratio(2 * precision * recall, precision + recall)),
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
    r"""Without the accuracy, which equals ``f1_micro`` and the plan's
    ``acc``."""
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
    return [
        _model('logloss', float(arrays['logloss']) / rows),
        _model('f1_macro', float(f1[seen].mean())),
        _model('f1_micro', hits.sum() / rows),
        _model('f1_weighted', float((f1 * support).sum() / rows)),
        Series('holdout/by_k/accuracy',
               tuple((k, float(arrays[f'top_{k}']) / rows)
                     for k in range(1, min(_TOP_K, size) + 1)), 'k'),
        _by_class('holdout/by_class', 'class', names, top, support,
                  precision, recall, f1),
        _confusion(confusion, names, top),
        _calibration(PROBABILITY_EDGES, (0.0, 1.0), arrays['confidence'],
                     arrays['confidence_sum'], arrays['correct'],
                     ('mean confidence', 'accuracy')),
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
    r"""With ``mae`` and ``rmse``, which the trainer's own metrics of the
    same names replace."""
    arrays = counts.arrays
    rows = float(arrays['rows'])
    total = float(arrays['target'])
    spread = float(arrays['target_squared']) - total * total / rows
    squared = float(arrays['squared_error'])
    edges = counts.edges['target']
    return [
        _model('mae', float(arrays['abs_error']) / rows),
        _model('rmse', float(np.sqrt(squared / rows))),
        _model('r2', 1 - squared / spread if spread > 0 else 0.0),
        _model('bias', float(arrays['error']) / rows),
        _model('mape', _ratio(float(arrays['ape']),
                              float(arrays['ape_rows']))),
        _model('smape', _ratio(float(arrays['sape']),
                               float(arrays['sape_rows']))),
        counts.summary('TARGET', 'target', 'holdout/histogram/actual'),
        counts.summary('TARGET_PRED', 'target', 'holdout/histogram/predicted'),
        counts.summary('residual', 'residual', 'holdout/residuals/rows'),
        _calibration(edges, counts.extents.get('TARGET_PRED'),
                     arrays['TARGET_PRED'], arrays['pred_sum'],
                     arrays['target_sum'], ('mean predicted', 'mean actual')),
    ]


def _multilabel(prob: np.ndarray, pred: np.ndarray,
                target: np.ndarray) -> Counts:
    size = prob.shape[1]
    counts = Counts({'score': PROBABILITY_EDGES,
                     'top_score': PROBABILITY_EDGES})
    for name, rows in (('tp', pred & target), ('fp', pred & ~target),
                       ('fn', ~pred & target), ('support', target)):
        counts.tally(name, np.nonzero(rows)[1], size)
    counts.tally('labels', target.sum(axis=1), size + 1)
    counts.tally('predicted_labels', pred.sum(axis=1), size + 1)
    counts.histogram('positive', 'score', prob[target])
    counts.histogram('negative', 'score', prob[~target])
    top = prob.max(axis=1)
    counts.histogram('top_score', 'top_score', top)
    counts.extent('top_score', top)
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
    entities = np.flatnonzero(arrays['labels'] + arrays['predicted_labels'])
    return [
        _model('f1_micro', _ratio(2 * tp_all, 2 * tp_all + fp_all + fn_all)),
        _model('precision_micro', _ratio(tp_all, tp_all + fp_all)),
        _model('recall_micro', _ratio(tp_all, tp_all + fn_all)),
        _model('f1_macro', float(f1[seen].mean()) if seen.any() else 0.0),
        _by_class('holdout/by_label', 'label', names, top, arrays['support'],
                  precision, recall, f1),
        _by_threshold(arrays['positive'], arrays['negative']),
        *(Series(f'holdout/by_label_count/{column}',
                 tuple((int(n), float(arrays[name][n])) for n in entities),
                 'labels')
          for name, column in (('labels', 'actual entities'),
                               ('predicted_labels', 'predicted entities'))),
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


def _horizon_summaries(counts: Counts,
                       timeframe: Optional[Timeframe]) -> List[Summary]:
    arrays = counts.arrays
    rows = np.maximum(arrays['horizon_rows'], 1)
    spread = (arrays['horizon_target_squared'] -
              arrays['horizon_target']**2 / rows)
    r2 = np.where(spread > 0,
                  1 - arrays['horizon_squared_error'] / np.maximum(spread,
                                                                   _EPS), 0.0)
    columns = {
        'rows': arrays['horizon_rows'].astype(np.int64),
        'mae': arrays['horizon_abs_error'] / rows,
        'rmse': np.sqrt(arrays['horizon_squared_error'] / rows),
        'bias': arrays['horizon_error'] / rows,
        'r2': r2,
    }
    steps = np.flatnonzero(arrays['horizon_rows'])
    window = () if timeframe is None else ('window',)
    return [
        Table('holdout/by_horizon', ('horizon', *window, *columns),
              tuple((int(h) + 1,
                     *(() if timeframe is None else
                       (horizon_label(int(h), timeframe),)),
                     *(values[h].item() for values in columns.values()))
                    for h in steps)),
        *(counts.summary(f'TARGET_PRED_h{h + 1}', 'target',
                         f'holdout/histogram/predicted, '
                         f'{horizon_label(h, timeframe)}')
          for h in sorted(int(name.removeprefix('TARGET_PRED_h')) - 1
                          for name in arrays
                          if name.startswith('TARGET_PRED_h'))),
    ]


def horizon_label(h: int, timeframe: Optional[Timeframe] = None) -> str:
    r"""Horizon ``h``, from 0, as its target window (``days 7–14``), or as
    ``horizon <h + 1>`` without a ``timeframe``."""
    if timeframe is None:
        return f'horizon {h + 1}'
    start, end, step, unit = timeframe
    return f'{unit} {start + h * step}–{end + h * step}'


def _horizons(size: int) -> List[int]:
    if size <= _HORIZONS:
        return list(range(size))
    return sorted({int(h) for h in np.linspace(0, size - 1, _HORIZONS)
                   .round()})


def _above(counts: np.ndarray) -> np.ndarray:
    r"""``counts`` at or above each edge: index ``k`` sums the bins from
    ``k + 1`` on."""
    return np.cumsum(counts[::-1])[::-1][1:].astype(np.float64)


def _by_threshold(positive: np.ndarray, negative: np.ndarray) -> Table:
    r"""The rows scored at or above each hundredth, and the confusion
    counts and rates of calling them positive."""
    tp, fp = _above(positive), _above(negative)
    positives, negatives = float(positive.sum()), float(negative.sum())
    return Table(
        'holdout/by_threshold',
        ('threshold', 'share at or above', 'tp', 'fp', 'fn', 'tn',
         'precision', 'recall', 'fpr'),
        tuple((t, _ratio(tp[i] + fp[i], positives + negatives), int(tp[i]),
               int(fp[i]), int(positives - tp[i]), int(negatives - fp[i]),
               _share(tp[i], tp[i] + fp[i]), _share(tp[i], positives),
               _share(fp[i], negatives))
              for i, t in enumerate(PROBABILITY_EDGES)))


def _calibration(edges: Sequence[float],
                 extent: Optional[Tuple[float, float]], rows: np.ndarray,
                 x_sum: np.ndarray, y_sum: np.ndarray,
                 names: Tuple[str, str]) -> Table:
    r"""Per non-empty bin, its rows and their mean ``x`` and ``y``."""
    bounds = bin_bounds(edges, extent)
    return Table('holdout/calibration', ('from', 'to', 'rows', *names),
                 tuple((bounds[b].item(), bounds[b + 1].item(), int(rows[b]),
                        (x_sum[b] / rows[b]).item(),
                        (y_sum[b] / rows[b]).item())
                       for b in np.flatnonzero(rows)))


def _by_class(name: str, label: str, names: Sequence[str],
              top: Sequence[int], support: np.ndarray, *metrics: np.ndarray
              ) -> Table:
    return Table(name, (label, 'support', 'precision', 'recall', 'f1'),
                 tuple((names[i], int(support[i]),
                        *(float(values[i]) for values in metrics))
                       for i in top))


def _confusion(confusion: np.ndarray, names: Sequence[str],
               top: Sequence[int]) -> Confusion:
    rest = [i for i in range(len(names)) if i not in set(top)]
    groups: List[Tuple[str, List[int]]] = [(names[i], [i]) for i in top]
    if rest:
        groups.append(('other', rest))
    return Confusion('holdout/confusion', tuple(name for name, _ in groups),
                     tuple(tuple(int(confusion[np.ix_(actual, predicted)].sum())
                                 for _, predicted in groups)
                           for _, actual in groups))


def _model(name: str, value: float) -> Scalar:
    return Scalar(name, float(value), role='model')


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _share(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def class_names(target_encoder: Any) -> List[Any]:
    mapping = getattr(target_encoder.tensor_mapper, 'mapping', None)
    return [] if mapping is None else list(mapping)
