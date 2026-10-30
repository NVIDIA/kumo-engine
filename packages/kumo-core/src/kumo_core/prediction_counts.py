# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact counts of a prediction run's scores, and the summaries a tracker
logs from them.

Each probability column (``True_PROB``; each class's, for at most
`MAX_CLASS_HISTOGRAMS` classes; and ``confidence``, the top class's
probability) is counted on the ``score`` axis, ``TARGET_PRED`` on the
``target`` axis, the training target's bins, and each entity's top score,
when the holdout counted it, on its ``top_score`` axis: a column ``c`` has
the counts ``c``, the sum ``c/sum`` and the extent ``c``. ``classes``
tallies each row's top class, ``predicted_positive`` the rows
``TARGET_PRED`` labels positive, ``label_positive`` the rows predicting each
label, ``predicted_labels`` the rows per number of predicted labels and
``items`` each catalogue item's recommendations.
"""

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from kumo_core.evaluation_counts import Timeframe, _horizons, horizon_label
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.score_drift import PSI_SHIFTS, axis, drift_column, score_drift
from kumo_core.summaries import Note, Scalar, Series, Summary, Table
from kumo_core.types import TaskType

TOP_CLASSES = 20
ITEMS = 'items'
MAX_CLASS_HISTOGRAMS = 100
_SCORES = ('confidence', 'top_score', 'TARGET_PRED')


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
        count_top_scores(counts, prob.max(axis=1))
    else:
        counts.tally('classes', prob.argmax(axis=1), prob.shape[1])
        _column(counts, 'confidence', 'score', prob.max(axis=1))
    if len(columns) <= MAX_CLASS_HISTOGRAMS:
        for index, name in enumerate(columns):
            if name != 'False_PROB':
                _column(counts, name, 'score', prob[:, index])


def count_top_scores(counts: Counts, scores: Any) -> None:
    r"""Adds each entity's top score of one batch, if ``counts`` has the
    holdout's ``top_score`` bins."""
    if 'top_score' in counts.edges:
        _column(counts, 'top_score', 'top_score', scores)


def count_recommendations(counts: Counts, items: Any, catalogue: int) -> None:
    r"""Adds one batch of recommended ``items``, indices into a catalogue of
    ``catalogue`` items, to the ``items`` tally; out-of-range indices are
    padding."""
    items = np.asarray(items, dtype=np.int64).reshape(-1)
    counts.tally(ITEMS, items[(items >= 0) & (items < catalogue)], catalogue)


def item_counts(counts: Counts) -> Counts:
    r"""``counts`` with its ``items`` tally replaced by what is logged of
    it, which is as small for any catalogue: ``item_frequency``, items per
    number of recommendations on powers of two, and ``item_frequency/sum``,
    their recommendations; and the totals ``catalogue``,
    ``items_recommended``, ``recommendations``, ``most_recommendations``,
    those of the most recommended item, and ``top1pct_recommendations``,
    those of the most recommended 1% of items."""
    arrays = dict(counts.arrays)
    tally = arrays.pop(ITEMS)
    top = max(1, math.ceil(tally.size / 100))
    edges = tuple(2.0**power for power in range(
        int(np.log2(max(tally.max(initial=1), 1))) + 1))
    summary = Counts({**counts.edges, 'recommendations': edges}, arrays,
                     dict(counts.extents))
    summary.histogram('item_frequency', 'recommendations', tally)
    summary.histogram('item_frequency/sum', 'recommendations', tally,
                      weights=tally)
    for name, value in (('catalogue', tally.size),
                        ('items_recommended', np.count_nonzero(tally)),
                        ('recommendations', tally.sum()),
                        ('most_recommendations', tally.max(initial=0)),
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
                         baseline: Optional[Counts],
                         timeframe: Optional[Timeframe] = None
                         ) -> List[Summary]:
    r"""``pred/histogram`` of each score column, of only the `TOP_CLASSES`
    most predicted classes, and of each forecast horizon; ``pred/summary``,
    the mean, minimum and maximum of the score that drift is measured on;
    ``pred/positive_rate`` at a binary ``threshold``; ``pred/by_class``
    and, for labels, ``pred/by_label_count``; for recommendations,
    ``pred/item_frequency``, ``pred/item_coverage`` and
    ``pred/top_1pct_items_share``; and ``drift/psi``, ``drift/ks`` and
    ``drift/wasserstein`` against ``baseline``, the training holdout's
    counts, when it counted the same score on the same bins. A forecast's
    horizons are named by their window in ``timeframe``, as the holdout's
    are."""
    summaries: List[Summary] = []
    scores = [name for name in counts.extents if name in _SCORES]
    columns = [name for name in counts.extents if name not in _SCORES]
    ranked = next((name for name in ('classes', 'label_positive')
                   if name in counts.arrays), None)
    if ranked is not None:
        order = np.argsort(-counts.arrays[ranked], kind='stable').tolist()
        summaries.append(_by_class(counts, ranked, order, columns, baseline))
        columns = [columns[i] for i in order[:TOP_CLASSES] if columns]
    summaries += [
        counts.summary(name, axis(name), f'pred/histogram/{_label(name)}')
        for name in (*scores, *columns)
    ]
    summaries += [
        counts.summary(name, 'target', 'pred/histogram/predicted, ' +
                       horizon_label(int(name.removeprefix('TARGET_PRED_h'))
                                     - 1, timeframe))
        for name in counts.arrays if name.startswith('TARGET_PRED_h')
    ]
    score = drift_column(counts)
    if score is not None and score in counts.extents:
        low, high = counts.extents[score]
        summaries.append(Table(
            'pred/summary', ('column', 'mean', 'min', 'max'),
            ((_label(score), float(counts.arrays[f'{score}/sum'] /
                                   counts.arrays[score].sum()), low, high),),
            brief=True))
    if threshold is not None and 'predicted_positive' in counts.arrays:
        summaries.append(Scalar(
            'pred/positive_rate', float(counts.arrays['predicted_positive'] /
                                        counts.arrays['True_PROB'].sum())))
    if ranked == 'label_positive':
        per_row = counts.arrays['predicted_labels']
        summaries.append(Series(
            'pred/by_label_count/entities',
            tuple((int(n), float(per_row[n])) for n in np.flatnonzero(per_row)),
            'labels'))
    if 'item_frequency' in counts.arrays:
        summaries += _item_summaries(counts)
    drift = None if baseline is None else score_drift(counts, baseline)
    if drift is not None:
        summaries += [Scalar(f'drift/{name}', value)
                      for name, value in drift.items()]
        summaries.append(Note('notes', _drift_note(counts, score)))
    return summaries


def _by_class(counts: Counts, ranked: str, order: Sequence[int],
              columns: Sequence[str], baseline: Optional[Counts]) -> Table:
    r"""Each class's or label's share of the rows predicting it, here and
    at the ``baseline`` holdout when it tallied the same classes, and the
    mean, minimum and maximum of its probability, if counted."""
    tally = counts.arrays[ranked]
    rows = (tally.sum() if ranked == 'classes' else
            counts.arrays['predicted_labels'].sum())
    holdout = None if baseline is None else baseline.arrays.get(ranked)
    if holdout is not None and (holdout.shape != tally.shape
                                or not holdout.sum()):
        holdout = None

    def stats(name: Optional[str]) -> tuple:
        if name is None:
            return (None, None, None)
        low, high = counts.extents[name]
        return (float(counts.arrays[f'{name}/sum'] /
                      counts.arrays[name].sum()), low, high)

    label = 'class' if ranked == 'classes' else 'label'
    return Table(
        'pred/by_class', (label, 'share predicted', 'share at holdout',
                          'mean', 'min', 'max'),
        tuple((columns[i].removesuffix('_PROB') if columns else str(i),
               float(tally[i] / rows) if rows else 0.0,
               None if holdout is None else float(holdout[i] / holdout.sum()),
               *stats(columns[i] if columns else None)) for i in order))


def _item_summaries(counts: Counts) -> List[Summary]:
    arrays = counts.arrays
    total = {name: float(arrays[name])
             for name in ('catalogue', 'items_recommended', 'recommendations',
                          'most_recommendations', 'top1pct_recommendations')}
    edges = counts.edges['recommendations']
    bounds = [0, *(int(edge) for edge in edges),
              int(total['most_recommendations']) + 1]
    return [
        Table('pred/item_frequency',
              ('times recommended', 'items', 'share of recommendations'),
              tuple((_range(bounds[b], bounds[b + 1] - 1),
                     int(arrays['item_frequency'][b]),
                     float(arrays['item_frequency/sum'][b] /
                           max(total['recommendations'], 1)))
                    for b in range(len(edges) + 1))),
        Scalar('pred/item_coverage',
               total['items_recommended'] / max(total['catalogue'], 1)),
        Scalar('pred/top_1pct_items_share', total['top1pct_recommendations'] /
               max(total['recommendations'], 1)),
    ]


def _range(low: int, high: int) -> str:
    return str(low) if high <= low else f'{low}–{high}'


def _drift_note(counts: Counts, score: str) -> str:
    bins = ('100 bins of width 0.01'
            if counts.edges[axis(score)] == PROBABILITY_EDGES else
            "the training target's histogram bins" if score == 'TARGET_PRED'
            else "percentile bins of the holdout's top scores")
    rule = ', '.join(f'{bound} {shift}' for shift, bound in PSI_SHIFTS.items())
    return (f"Score drift compares `{_label(score)}` with the model's holdout "
            f'scores at training, over {bins}. PSI rule of thumb: 0 '
            f'identical, {rule} shift. KS: the largest gap between the two '
            'cumulative distributions, from 0 to 1. Wasserstein: how far the '
            'scores moved on average, in score units.')


def _label(column: str) -> str:
    r"""A score column's name as a table column."""
    return {'TARGET_PRED': 'predicted', 'top_score': 'top score'}.get(
        column, column.replace('/', '_'))
