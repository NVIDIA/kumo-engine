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
each row's top class and ``predicted_positive`` the rows ``TARGET_PRED``
labels positive.
"""

from typing import Any, List, Optional, Sequence

import numpy as np
from kumo_core.score_counts import Counts
from kumo_core.summaries import Scalar, Summary, Table
from kumo_core.types import TaskType

QUANTILES = (1, 5, 25, 50, 75, 95, 99)
TOP_CLASSES = 20
MAX_CLASS_HISTOGRAMS = 100
_EPSILON = 1e-6


def count_predictions(counts: Counts, task_type: TaskType, *, prob: Any,
                      pred: Any, target_pred: Any,
                      columns: Sequence[str]) -> None:
    r"""Adds one batch: ``prob`` and ``pred`` as the model decoded them,
    ``target_pred`` the written ``TARGET_PRED`` and ``columns`` the
    probability columns' names, in class order."""
    if task_type.has_numerical_target:
        _column(counts, 'TARGET_PRED', 'target', target_pred)
        return
    prob = np.asarray(prob, dtype=np.float64)
    if task_type is TaskType.BINARY_CLASSIFICATION:
        counts.total('predicted_positive',
                     np.count_nonzero(np.asarray(pred, dtype=bool)))
    else:
        counts.tally('classes', prob.argmax(axis=1), prob.shape[1])
        _column(counts, 'confidence', 'score', prob.max(axis=1))
    if len(columns) <= MAX_CLASS_HISTOGRAMS:
        for index, name in enumerate(columns):
            _column(counts, name, 'score', prob[:, index])


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
    binary ``threshold``; ``pred/classes``, each class's rows and share;
    and ``drift/score_psi`` and ``drift/score_ks`` against ``baseline``,
    the training holdout's counts."""
    columns = list(counts.extents)
    if 'classes' in counts.arrays:
        tally = counts.arrays['classes']
        labels = [name for name in columns if name != 'confidence']
        top = np.argsort(-tally, kind='stable')
        columns = [
            'confidence', *(labels[i] for i in top[:TOP_CLASSES] if labels)
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
    if 'classes' in counts.arrays:
        summaries.append(_class_shares(tally, top, labels))
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
    r"""Where a score column's summaries go: ``pred/<column>``, or
    ``pred`` for ``TARGET_PRED``."""
    return 'pred' if column == 'TARGET_PRED' else f'pred/{column}'


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
    return np.interp(np.array(QUANTILES) / 100, cdf, bounds).tolist()


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
