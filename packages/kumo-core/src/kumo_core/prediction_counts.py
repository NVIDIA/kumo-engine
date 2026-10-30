# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact counts of a prediction run's scores, and the summaries a tracker
logs from them.

Each probability column (``False_PROB`` and ``True_PROB``; each class's,
for at most `MAX_CLASS_HISTOGRAMS` classes; and ``confidence``, the top
class's probability) is counted on the ``score`` axis, and ``TARGET_PRED``
on the ``target`` axis, the training target's bins: a column ``c`` has the
counts ``c``, the sum ``c/sum`` and the extent ``c``. ``classes`` tallies
each row's top class, ``predicted_positive`` the rows ``TARGET_PRED``
labels positive, ``label_positive`` the rows predicting each label,
``predicted_labels`` the rows per number of predicted labels and ``items``
each catalogue item's recommendations.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from kumo_core.evaluation_counts import _horizons
from kumo_core.score_counts import Counts
from kumo_core.summaries import Scalar, Series, Summary, Table
from kumo_core.types import TaskType

QUANTILES = (1, 5, 25, 50, 75, 95, 99)
TOP_CLASSES = 20
ITEMS = 'items'
MAX_CLASS_HISTOGRAMS = 100
_EPSILON = 1e-6


def count_predictions(counts: Counts, task_type: TaskType, *, prob: Any,
                      pred: Any, target_pred: Any, columns: Sequence[str],
                      horizon: Any = None) -> None:
    r"""Adds one batch: ``prob`` and ``pred`` as the model decoded them,
    ``target_pred`` the written ``TARGET_PRED``, ``columns`` the
    probability columns' names, in class order, and ``horizon`` each
    forecast row's step ahead, from 0, whose `_horizons` are also counted
    alone as ``TARGET_PRED_h<step>``."""
    if task_type.has_numerical_target:
        _column(counts, 'TARGET_PRED', 'target', target_pred)
        if horizon is not None:
            values = np.asarray(target_pred, dtype=np.float64)
            horizon = np.asarray(horizon, dtype=np.int64)
            for h in _horizons(int(horizon.max()) + 1 if horizon.size else 0):
                counts.histogram(f'TARGET_PRED_h{h + 1}', 'target',
                                 values[horizon == h])
        return
    prob = np.asarray(prob, dtype=np.float64)
    if task_type is TaskType.BINARY_CLASSIFICATION:
        counts.total('predicted_positive',
                     np.count_nonzero(np.asarray(pred, dtype=bool)))
    elif task_type is TaskType.MULTILABEL_CLASSIFICATION:
        positive = np.asarray(pred, dtype=bool)
        counts.tally('label_positive', np.nonzero(positive)[1], prob.shape[1])
        counts.tally('predicted_labels', positive.sum(axis=1),
                     prob.shape[1] + 1)
    else:
        counts.tally('classes', prob.argmax(axis=1), prob.shape[1])
        _column(counts, 'confidence', 'score', prob.max(axis=1))
    if len(columns) <= MAX_CLASS_HISTOGRAMS:
        for index, name in enumerate(columns):
            _column(counts, name, 'score', prob[:, index])


def count_recommendations(counts: Counts, items: Any, catalogue: int) -> None:
    r"""Adds one batch of recommended ``items``, indices into a catalogue of
    ``catalogue`` items, to the ``items`` tally; out-of-range indices are
    padding."""
    items = np.asarray(items, dtype=np.int64).reshape(-1)
    counts.tally(ITEMS, items[(items >= 0) & (items < catalogue)], catalogue)


def item_counts(counts: Counts) -> Counts:
    r"""``counts`` with its ``items`` tally replaced by what is logged of
    it, which is as small for any catalogue: ``item_frequency``, items per
    number of recommendations on powers of two, and the totals
    ``catalogue``, ``items_recommended``, ``recommendations`` and
    ``top1pct_recommendations``, those of the most recommended 1% of
    items."""
    arrays = dict(counts.arrays)
    tally = arrays.pop(ITEMS)
    top = max(1, math.ceil(tally.size / 100))
    edges = tuple(2.0**power for power in range(
        int(np.log2(max(tally.max(initial=1), 1))) + 1))
    summary = Counts({**counts.edges, 'recommendations': edges}, arrays,
                     dict(counts.extents))
    summary.histogram('item_frequency', 'recommendations', tally)
    for name, value in (('catalogue', tally.size),
                        ('items_recommended', np.count_nonzero(tally)),
                        ('recommendations', tally.sum()),
                        ('top1pct_recommendations',
                         np.partition(tally, tally.size - top)[-top:].sum()
                         if tally.size else 0)):
        summary.total(name, value)
    return summary


def save_counts(counts: Counts, path: Path) -> None:
    r"""Writes ``counts`` as JSON at ``path``, and its ``items`` tally, as
    large as the catalogue, as NumPy beside it."""
    arrays = dict(counts.arrays)
    items = arrays.pop(ITEMS, None)
    if items is not None:
        np.save(path.with_suffix('.npy'), items)
    path.write_text(json.dumps(
        Counts(counts.edges, arrays, counts.extents).to_dict()))


def load_counts(path: Path) -> Counts:
    counts = Counts.from_dict(json.loads(path.read_text()))
    if path.with_suffix('.npy').exists():
        counts.arrays[ITEMS] = np.load(path.with_suffix('.npy'))
    return counts


def _column(counts: Counts, name: str, axis: str, values: Any) -> None:
    values = np.asarray(values, dtype=np.float64)
    counts.histogram(name, axis, values)
    counts.total(f'{name}/sum', values[np.isfinite(values)].sum())
    counts.extent(name, values)


def prediction_summaries(counts: Counts, *, threshold: Optional[float],
                         baseline: Optional[Counts]) -> List[Summary]:
    r"""``pred/<column>/hist`` and its ``mean``, ``min``, ``max`` and
    quantiles (``pred/...`` for ``TARGET_PRED``) for each column, of only
    the `TOP_CLASSES` most predicted classes; ``pred/positive_rate`` at a
    binary ``threshold``; ``pred/h<step>/hist`` per forecast horizon;
    ``pred/classes``, each class's rows and share; for labels,
    ``pred/labels_per_entity``, rows per number of predicted labels, and
    ``pred/label/<label>/share``, the rows predicting each top label; for
    recommendations, ``pred/item_frequency/hist``, ``pred/item_coverage``
    and ``pred/top1pct_share``;
    and ``drift/score_psi`` and ``drift/score_ks`` against ``baseline``,
    the training holdout's counts."""
    columns = list(counts.extents)
    ranked = next((name for name in ('classes', 'label_positive')
                   if name in counts.arrays), None)
    if ranked is not None:
        tally = counts.arrays[ranked]
        labels = [name for name in columns if name != 'confidence']
        top = np.argsort(-tally, kind='stable')
        columns = [
            *(name for name in columns if name == 'confidence'),
            *(labels[i] for i in top[:TOP_CLASSES] if labels)
        ]
    summaries = [
        summary for name in columns
        for summary in _column_summaries(counts, name)
    ]
    if threshold is not None and 'predicted_positive' in counts.arrays:
        summaries.append(
            Scalar(
                'pred/positive_rate',
                float(counts.arrays['predicted_positive'] /
                      counts.arrays['True_PROB'].sum())))
    summaries += [
        counts.summary(name, 'target',
                       f"pred/{name.removeprefix('TARGET_PRED_')}/hist")
        for name in counts.arrays if name.startswith('TARGET_PRED_h')
    ]
    if ranked == 'classes':
        summaries.append(_class_shares(tally, top, labels))
    if ranked == 'label_positive':
        summaries += _label_summaries(counts, top[:TOP_CLASSES], labels)
    if 'item_frequency' in counts.arrays:
        summaries += _item_summaries(counts)
    score = next((name for name in ('True_PROB', 'confidence', 'TARGET_PRED')
                  if name in counts.extents), None)
    if score is not None and baseline is not None:
        summaries += _drift(counts, baseline, score)
    return summaries


def _class_shares(tally: np.ndarray, order: Sequence[int],
                  labels: Sequence[str]) -> Table:
    return Table(
        'pred/classes', ('class', 'rows', 'share'),
        tuple((labels[i].removesuffix('_PROB') if labels else str(i),
               int(tally[i]), tally[i] / tally.sum()) for i in order))


def _label_summaries(counts: Counts, top: Sequence[int],
                     labels: Sequence[str]) -> List[Summary]:
    per_row = counts.arrays['predicted_labels']
    rows = per_row.sum()
    last = int(np.flatnonzero(per_row).max()) if rows else 0
    names = [_metric_part(labels[i].removesuffix('_PROB')) if labels else
             str(i) for i in top]
    return [
        Series('pred/labels_per_entity',
               tuple((k, float(per_row[k])) for k in range(last + 1)),
               'labels'),
        *(Scalar(f'pred/label/{name}/share',
                 float(counts.arrays['label_positive'][i] / rows))
          for name, i in zip(names, top)),
    ]


def _item_summaries(counts: Counts) -> List[Summary]:
    total = {name: float(counts.arrays[name])
             for name in ('catalogue', 'items_recommended', 'recommendations',
                          'top1pct_recommendations')}
    return [
        counts.summary('item_frequency', 'recommendations',
                       'pred/item_frequency/hist'),
        Scalar('pred/item_coverage',
               total['items_recommended'] / max(total['catalogue'], 1)),
        Scalar('pred/top1pct_share', total['top1pct_recommendations'] /
               max(total['recommendations'], 1)),
    ]


def _drift(counts: Counts, baseline: Counts, score: str) -> List[Summary]:
    axis = _axis(score)
    if (score not in baseline.arrays
            or baseline.edges.get(axis) != counts.edges[axis]):
        return []
    actual, expected = counts.arrays[score], baseline.arrays[score]
    return [
        Scalar('drift/score_psi', psi(actual, expected)),
        Scalar('drift/score_ks', ks(actual, expected))
    ]


def summary_prefix(column: str) -> str:
    r"""Where a score column's summaries go: ``pred/<column>``, with
    characters metric names reject replaced by ``_``, or ``pred`` for
    ``TARGET_PRED``."""
    return 'pred' if column == 'TARGET_PRED' else f'pred/{_metric_part(column)}'


def _column_summaries(counts: Counts, name: str) -> List[Summary]:
    prefix = summary_prefix(name)
    axis = _axis(name)
    histogram = counts.arrays[name]
    low, high = counts.extents[name]
    values = quantiles(histogram, counts.edges[axis], low, high)
    return [
        counts.summary(name, axis, f'{prefix}/hist'),
        Scalar(f'{prefix}/mean',
               float(counts.arrays[f'{name}/sum'] / histogram.sum())),
        Scalar(f'{prefix}/min', low),
        Scalar(f'{prefix}/max', high),
        *(Scalar(f'{prefix}/q{q:02d}', value)
          for q, value in zip(QUANTILES, values)),
    ]


def _axis(name: str) -> str:
    return 'target' if name == 'TARGET_PRED' else 'score'


def quantiles(histogram: np.ndarray, edges: Sequence[float], low: float,
              high: float) -> List[float]:
    r"""`QUANTILES` of counted values from their ``histogram`` over
    ``edges``, linear within a bin, the end bins closed by the values'
    exact ``low`` and ``high``: each is within its bin of the exact
    quantile."""
    bounds = np.clip([low, *edges, high], low, high)
    cdf = np.concatenate([[0.0], np.cumsum(histogram)]) / histogram.sum()
    levels = np.array(QUANTILES) / 100
    reached = np.clip(np.searchsorted(cdf, levels), 1, len(cdf) - 1)
    below, above = cdf[reached - 1], cdf[reached]
    step = np.divide(levels - below, above - below, out=np.ones_like(levels),
                     where=above > below)
    return (bounds[reached - 1] + step *
            (bounds[reached] - bounds[reached - 1])).tolist()


def psi(actual: np.ndarray, expected: np.ndarray) -> float:
    r"""Population stability index of two histograms over the same bins,
    each bin's share at least ``1e-6``."""
    p, q = _shares(actual), _shares(expected)
    return float(np.sum((p - q) * np.log(p / q)))


def ks(actual: np.ndarray, expected: np.ndarray) -> float:
    r"""The largest gap between two histograms' cumulative shares, at the
    bin edges."""
    return float(
        np.max(
            np.abs(
                np.cumsum(actual) / actual.sum() -
                np.cumsum(expected) / expected.sum())))


def _shares(histogram: np.ndarray) -> np.ndarray:
    return np.maximum(histogram / histogram.sum(), _EPSILON)


def _metric_part(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.\- ]', '_', name)
