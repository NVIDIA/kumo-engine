# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact counts of a prediction run's scores, and their summaries."""

import json
import math
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from kumo_core.evaluation_counts import (
    TOP_CLASSES,
    Timeframe,
    horizon_label,
    kept_horizons,
)
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.score_drift import PSI_SHIFTS, axis, drift_column, score_drift
from kumo_core.summaries import (
    BarChart,
    Note,
    Scalar,
    Series,
    Summary,
    Table,
)
from kumo_core.types import TaskType

ITEMS = 'items'
ENTITIES = 'items/entities'
_UINT32 = np.iinfo(np.uint32).max
_CHUNK = 1 << 24
RANKED_ITEMS = 100
MAX_CLASS_HISTOGRAMS = 100
_SCORES = ('confidence', 'top_score', 'TARGET_PRED')


def count_predictions(counts: Counts, task_type: TaskType, *, prob: Any,
                      pred: Any, target_pred: Any, columns: Sequence[str],
                      horizon: Any = None) -> None:
    r"""Adds one batch of decoded scores."""
    if task_type.has_numerical_target:
        _column(counts, 'TARGET_PRED', 'target', target_pred)
        if horizon is not None:
            values = np.asarray(target_pred, dtype=np.float64)
            horizon = np.asarray(horizon, dtype=np.int64)
            for h in kept_horizons(int(horizon.max()) + 1 if horizon.size else 0):
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
    r"""Adds each entity's top score, on the holdout's top_score bins."""
    if 'top_score' in counts.edges:
        _column(counts, 'top_score', 'top_score', scores)


def count_recommendations(counts: Counts, items: Any, catalogue: int) -> None:
    r"""Adds each entity's recommended items; out-of-range ones are padding."""
    items = np.asarray(items, dtype=np.int64)
    counts.total(ENTITIES, len(items))
    dtype = tally_dtype(float(counts.arrays[ENTITIES]))
    tally = counts.arrays.get(ITEMS)
    if tally is None:
        tally = counts.arrays[ITEMS] = np.zeros(catalogue, dtype)
    if tally.size != catalogue:
        raise ValueError(f'{ITEMS} has {tally.size} items, not {catalogue}')
    if tally.dtype != dtype:
        tally = counts.arrays[ITEMS] = tally.astype(dtype)
    items = items.reshape(-1)
    np.add.at(tally, items[(items >= 0) & (items < catalogue)], 1)


def tally_dtype(entities: float) -> type:
    r"""uint32 while no item can be recommended more often."""
    return np.uint32 if entities <= _UINT32 else np.uint64


def item_counts(counts: Counts) -> Counts:
    r"""The items tally as small frequency and rank counts."""
    arrays = dict(counts.arrays)
    tally = arrays.pop(ITEMS)
    arrays.pop(ENTITIES, None)
    most = int(tally.max(initial=0))
    edges = tuple(2.0**power for power in range(
        int(np.log2(max(most, 1))) + 1))
    summary = Counts({**counts.edges, 'recommendations': edges}, arrays,
                     dict(counts.extents))
    top = max(1, math.ceil(tally.size / 100))
    ranked = top1pct = np.zeros(0, tally.dtype)
    for start in range(0, max(tally.size, 1), _CHUNK):
        chunk = tally[start:start + _CHUNK]
        summary.histogram('item_frequency', 'recommendations', chunk)
        summary.histogram('item_frequency/sum', 'recommendations', chunk,
                          weights=chunk)
        ranked = _largest(np.concatenate([ranked, chunk]), RANKED_ITEMS)
        top1pct = _largest(np.concatenate([top1pct, chunk]), top)
    summary.arrays['items_by_rank'] = np.sort(ranked)[::-1]
    for name, value in (('catalogue', tally.size),
                        ('items_recommended', np.count_nonzero(tally)),
                        ('recommendations', tally.sum()),
                        ('most_recommendations', most),
                        ('top1pct_recommendations', top1pct.sum())):
        summary.total(name, value)
    return summary


def _largest(values: np.ndarray, k: int) -> np.ndarray:
    return values if values.size <= k else np.partition(values, -k)[-k:]


def save_counts(counts: Counts, path: Path) -> None:
    r"""Writes counts as JSON, and the items tally as NumPy beside it."""
    arrays = dict(counts.arrays)
    items = arrays.pop(ITEMS, None)
    if items is not None:
        np.save(path.with_suffix('.npy'), items)
    path.write_text(json.dumps(
        Counts(counts.edges, arrays, counts.extents).to_dict()))


def _column(counts: Counts, name: str, edges: str, values: Any) -> None:
    values = np.asarray(values, dtype=np.float64)
    counts.histogram(name, edges, values)
    counts.total(f'{name}/sum', values[np.isfinite(values)].sum())
    counts.extent(name, values)


def prediction_summaries(counts: Counts, *, threshold: Optional[float],
                         baseline: Optional[Counts],
                         timeframe: Optional[Timeframe] = None,
                         entity: str = 'entities',
                         as_of: Optional[str] = None) -> List[Summary]:
    r"""Score histograms, shares and drift vs the holdout."""
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
    for name in (*scores, *columns):
        if counts.edges[axis(name)] == PROBABILITY_EDGES:
            summaries += _score_histogram(name, counts, baseline, entity,
                                          as_of)
    if 'TARGET_PRED' in counts.extents:
        summaries.append(_prediction_histogram(counts, baseline, entity,
                                               as_of))
    summaries += [
        counts.summary(name, 'target', 'pred/histogram/predicted, ' +
                       horizon_label(int(name.removeprefix('TARGET_PRED_h'))
                                     - 1, timeframe))
        for name in counts.arrays if name.startswith('TARGET_PRED_h')
    ]
    score = drift_column(counts)
    if score is not None and score in counts.extents:
        summaries.append(Table(
            'pred/summary', ('column', 'mean', 'min', 'max'),
            ((_label(score), *_mean_range(counts, score)),), brief=True))
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
        summaries += _item_summaries(counts, as_of)
    drift = None if baseline is None else score_drift(counts, baseline)
    if drift is not None:
        summaries += [Scalar(f'drift/{name}', value)
                      for name, value in drift.items()]
        summaries.append(Note('drift', _drift_note(counts, score)))
    return summaries


def _by_class(counts: Counts, ranked: str, order: Sequence[int],
              columns: Sequence[str], baseline: Optional[Counts]) -> Table:
    tally = counts.arrays[ranked]
    rows = (tally.sum() if ranked == 'classes' else
            counts.arrays['predicted_labels'].sum())
    holdout = None if baseline is None else baseline.arrays.get(ranked)
    if holdout is not None and (holdout.shape != tally.shape
                                or not holdout.sum()):
        holdout = None
    label = 'class' if ranked == 'classes' else 'label'
    return Table(
        'pred/by_class', (label, 'share now', 'holdout share', 'mean prob',
                          'min prob', 'max prob'),
        tuple((columns[i].removesuffix('_PROB') if columns else str(i),
               float(tally[i] / rows) if rows else 0.0,
               None if holdout is None else float(holdout[i] / holdout.sum()),
               *(_mean_range(counts, columns[i]) if columns else
                 (None, None, None))) for i in order))


def _mean_range(counts: Counts, name: str) -> Tuple[float, float, float]:
    low, high = counts.extents[name]
    return (float(counts.arrays[f'{name}/sum'] / counts.arrays[name].sum()),
            low, high)


def _item_summaries(counts: Counts, as_of: Optional[str]) -> List[Summary]:
    arrays = counts.arrays
    total = {name: float(arrays[name])
             for name in ('catalogue', 'items_recommended', 'recommendations',
                          'most_recommendations', 'top1pct_recommendations')}
    coverage = total['items_recommended'] / max(total['catalogue'], 1)
    top_share = (total['top1pct_recommendations'] /
                 max(total['recommendations'], 1))
    title = (f"Item popularity{f', {as_of}' if as_of else ''}: "
             f'{_percent(coverage)} of items recommended, top 1% get '
             f'{_percent(top_share)} of recommendations')
    edges = counts.edges['recommendations']
    bounds = [0, *(int(edge) for edge in edges),
              int(total['most_recommendations']) + 1]
    return [
        Table('pred/item_frequency',
              ('× recommended', 'items', 'share of recs'),
              tuple((_range(bounds[b], bounds[b + 1] - 1),
                     int(arrays['item_frequency'][b]),
                     float(arrays['item_frequency/sum'][b] /
                           max(total['recommendations'], 1)))
                    for b in range(len(edges) + 1))),
        Scalar('pred/item_coverage', coverage),
        Scalar('pred/top_1pct_items_share', top_share),
        *_item_popularity(arrays.get('items_by_rank', ()), title),
    ]


def _score_histogram(column: str, counts: Counts,
                     baseline: Optional[Counts], entity: str,
                     as_of: Optional[str]) -> List[Summary]:
    per_pct = _per_pct(counts.arrays[column])
    holdout = _holdout(baseline, column, PROBABILITY_EDGES)
    name, label = 'pred/score_histogram', ''
    if column != 'True_PROB':
        label = _label(column).removesuffix('_PROB')
        name += '/' + re.sub(r'[^\w.-]', '_', label)
    title = (f'Scores of {int(per_pct.sum()):,} {entity}' +
             (f' ({label})' if label else ''))
    return [
        Series(name, tuple(enumerate(per_pct.tolist())), 'score_pct',
               chart=True),
        _chart(name, title, ('score', entity), PROBABILITY_EDGES, per_pct,
               None if holdout is None else _per_pct(holdout), as_of),
    ]


def _prediction_histogram(counts: Counts, baseline: Optional[Counts],
                          entity: str, as_of: Optional[str]) -> BarChart:
    edges = counts.edges['target']
    bins = counts.arrays['TARGET_PRED']
    holdout = _holdout(baseline, 'TARGET_PRED', edges)
    lows, highs = zip(*(
        side.extents['TARGET_PRED'] for side in (counts, baseline)
        if side is not None and 'TARGET_PRED' in side.extents and
        (side is counts or holdout is not None)))
    return _chart(
        'pred/prediction_histogram',
        f'Predictions for {int(bins.sum()):,} {entity}',
        ("predicted value, on the training target's bins", entity),
        (min(*edges, *lows), *edges, max(*edges, *highs)), bins, holdout,
        as_of)


def _holdout(baseline: Optional[Counts], column: str,
             edges: Sequence[float]) -> Optional[np.ndarray]:
    if baseline is None or baseline.edges.get(axis(column)) != edges:
        return None
    holdout = baseline.arrays.get(column)
    return holdout if holdout is not None and holdout.sum() else None


def _chart(name: str, title: str, axes: Tuple[str, str],
           edges: Sequence[float], counts: np.ndarray,
           holdout: Optional[np.ndarray], as_of: Optional[str]) -> BarChart:
    line = () if holdout is None else tuple(
        (holdout * counts.sum() / holdout.sum()).tolist())
    title += (f', {as_of}' if as_of else '') + (' vs holdout' if line else '')
    return BarChart(name, title, axes, tuple(edges),
                    tuple(counts.astype(np.float64).tolist()), line,
                    ('this run', 'holdout, scaled to the same total'))


def _per_pct(counts: np.ndarray) -> np.ndarray:
    per_pct = counts[1:-1].astype(np.float64)
    per_pct[0] += counts[0]
    per_pct[-1] += counts[-1]
    return per_pct


def _item_popularity(ranked: Sequence[float], title: str) -> List[Summary]:
    points = tuple((rank, float(count))
                   for rank, count in enumerate(ranked, 1) if count)
    if len(points) < 2:
        return []
    return [
        Series('pred/item_popularity', points, 'item_rank', chart=True),
        BarChart('pred/item_popularity', title,
                 ('item rank', 'times recommended'),
                 tuple(float(edge) for edge in range(len(ranked) + 1)),
                 tuple(float(count) for count in ranked)),
    ]


def _percent(share: float) -> str:
    if share >= 0.01 or not share:
        return f'{share:.0%}'
    return f'{100 * share:.1g}%'


def _range(low: int, high: int) -> str:
    return str(low) if high <= low else f'{low}–{high}'


def _drift_note(counts: Counts, score: str) -> str:
    bins = ('100 bins of width 0.01'
            if counts.edges[axis(score)] == PROBABILITY_EDGES else
            "the training target's histogram bins" if score == 'TARGET_PRED'
            else "percentile bins of the holdout's top scores")
    rule = ', '.join(f'{bound} {shift}' for shift, bound in PSI_SHIFTS.items())
    reading = (
        'For link prediction the holdout covers only entities with a future '
        'target and a prediction all entities, so compare PSI with the '
        'previous run.' if 'item_frequency' in counts.arrays else
        f'PSI rule of thumb: 0 identical, {rule} shift.')
    return (f"Score drift compares `{_label(score)}` with the model's holdout "
            f"scores at training, on {bins}. PSI groups those bins at the "
            f"holdout's deciles, into up to 10 groups. {reading} KS: the "
            'largest gap between the two cumulative distributions, from 0 to '
            '1. Wasserstein: how far the scores moved on average, in score '
            'units.')


def _label(column: str) -> str:
    return {'TARGET_PRED': 'predicted', 'top_score': 'top score'}.get(
        column, column.replace('/', '_'))
