# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression card: Predicted against actual, from the published model's
holdout predictions and the training table's label histogram bin width."""

import math
import posixpath
from typing import Any, Optional

import numpy as np

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._run import (
    REGRESSION,
    Run,
    equal_count_groups,
    mapping,
    tenths,
)

ZERO_BAND_SHARE = 0.10
BANDS = 5
# Trailing histogram bins holding this many rows or fewer fold into one
# final bin, the boundary of the Kumo UI's Hide Outliers control.
FOLD_AT_MOST = 10
STATS = ('min', '25th percentile', 'median', 'mean', '75th percentile', 'max')


def _bands(run: Run, actual: np.ndarray, pred: np.ndarray,
           entities: np.ndarray) -> dict[str, Any]:
    n = len(actual)
    band = np.full(n, -1)
    zero = actual == 0
    if zero.mean() >= ZERO_BAND_SHARE:
        band[zero] = 0
        rest, parts, first = np.where(~zero)[0], BANDS - 1, 1
    else:
        rest, parts, first = np.arange(n), BANDS, 0
    if len(rest):
        band[rest] = first + equal_count_groups(actual[rest], entities[rest],
                                                parts)
    error = pred - actual
    squared = error**2
    total = float(squared.sum())
    rows, shares = [], []
    for b in range(first + parts if len(rest) else first):
        mask = band == b
        if not mask.any():
            continue
        if b >= first:
            low, high = fmt.numbers([actual[mask].min(), actual[mask].max()])
            label = f'{low} to {high}'
        else:
            label = '0'
        shares.append(float(squared[mask].sum()) / total if total else 0.0)
        median = float(np.median(error[mask]))
        rows.append(
            dict(label=label, median=median, median_text=fmt.signed(median),
                 error_share_text=fmt.percent(shares[-1])))
    worst = int(np.argmax(shares))
    return dict(id='bands', label='By actual value', x='actual value',
                y='median of predicted minus actual', bands=rows,
                highlight=worst,
                share_callout=f'{fmt.percent(shares[worst])} of squared error')


def bin_width(run: Run) -> Optional[float]:
    histogram = mapping(
        mapping(run.splits['test'].get('target_distribution')).get('histogram'))
    width = histogram.get('bin_width')
    if isinstance(width, (int, float)) and not isinstance(
            width, bool) and math.isfinite(width) and width > 0:
        return float(width)
    return None


def _edges(width: float, kept: int, folded: bool, start: int = 0) -> list[str]:
    edges = fmt.numbers([(start + i) * width for i in range(kept + 1)])
    labels = [f'{edges[i]} to {edges[i + 1]}' for i in range(kept)]
    if folded:
        labels.append(f'≥ {edges[kept]}')
    return labels


def _heatmap(run: Run, width: Optional[float]) -> Optional[dict[str, Any]]:
    from kumo_core.io.json import load_json
    from kumo_core.train.output import EvalOutput, TargetPredictionHeatMap

    trial = run.explanations.trial_root
    try:
        heatmap = None if trial is None else load_json(
            posixpath.join(trial, 'test_target_prediction_heatmap',
                           'heatmap.json'), TargetPredictionHeatMap)
    except (OSError, ValueError):
        heatmap = None
    if heatmap is not None:
        # Bounds that cannot be sized on a finite grid fall back to the
        # holdout instead of failing the widget.
        low = min(heatmap.label_bin_start, heatmap.prediction_bin_start)
        sizes = [heatmap.label_bin_end - low, heatmap.prediction_bin_end - low]
        if not (heatmap.cell_width > 0 and min(sizes) >= 0
                and sum(heatmap.counts) > 0 and np.isfinite([
                    *sizes, heatmap.cell_width, *np.ravel(heatmap.cells)
                ]).all()):
            heatmap = None
    coverage = None
    if heatmap is None:
        # Match the producer's 10,000-row limit; keep reopened views stable.
        if run.holdout is None or width is None:
            return None
        frame = run.holdout[['TARGET', 'TARGET_PRED']].copy()
        frame = frame[np.isfinite(frame['TARGET'])
                      & np.isfinite(frame['TARGET_PRED'])]
        if frame.empty:
            return None
        sample = frame.sample(min(len(frame), 10_000), random_state=0)
        heatmap = EvalOutput._generate_prediction_heatmap(
            sample, 'TARGET', 'TARGET_PRED', width, '')
        coverage = (f'Based on {len(sample):,} of '
                    f'{len(frame):,} retained holdout rows.')
    # Size the aligned axes before allocating, including constant-axis cases.
    start = min(heatmap.label_bin_start, heatmap.prediction_bin_start)
    x_span = heatmap.label_bin_end - start
    y_span = heatmap.prediction_bin_end - start
    cell_width = max(heatmap.cell_width,
                     math.sqrt(x_span) * math.sqrt(y_span) / 100,
                     max(x_span, y_span) / 9999)
    while True:
        columns = math.floor(x_span / cell_width + 1e-9) + 1
        rows = math.floor(y_span / cell_width + 1e-9) + 1
        if columns * rows <= 10_000:
            break
        cell_width *= max(1.01, math.sqrt(columns * rows / 10_000))
    cells = np.zeros((rows, columns), dtype=int)
    for (x, y), count in zip(heatmap.cells, heatmap.counts):
        row = min(rows - 1, max(0, round((y - start) / cell_width)))
        col = min(columns - 1, max(0, round((x - start) / cell_width)))
        cells[row, col] += count
    total = sum(heatmap.counts)

    def view(matrix: np.ndarray) -> dict[str, Any]:
        x_edges = [start + i * cell_width for i in range(matrix.shape[1] + 1)]
        y_edges = [start + i * cell_width for i in range(matrix.shape[0] + 1)]
        row, col = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        return dict(
            columns=[
                f'{fmt.number(a)} to {fmt.number(b)}'
                for a, b in zip(x_edges, x_edges[1:])
            ], rows=[
                f'{fmt.number(a)} to {fmt.number(b)}'
                for a, b in zip(y_edges, y_edges[1:])
            ], x_edges=x_edges, y_edges=y_edges, width=cell_width,
            counts=matrix.tolist(), densest=[int(row), int(col)],
            densest_callout=(f'{matrix[row, col]:,} {run.entities} '
                             f'({fmt.percent(matrix[row, col] / total)})'))

    def kept(counts: np.ndarray) -> int:
        indices = np.flatnonzero(counts > FOLD_AT_MOST)
        return int(indices[-1] + 1) if len(indices) else len(counts)

    trimmed = cells[:kept(cells.sum(axis=1)), :kept(cells.sum(axis=0))]
    return dict(id='heatmap',
                label='Heatmap', x='actual value', y='predicted value',
                **view(cells), trimmed=view(trimmed), coverage=coverage,
                diagonal='predicted = actual')


def _stats(values: np.ndarray) -> dict[str, Any]:
    numbers = (values.min(), np.percentile(values, 25), np.median(values),
               values.mean(), np.percentile(values, 75), values.max())
    texts = fmt.numbers(map(float, numbers))
    return dict(values=dict(max=float(values.max())),
                items=[dict(label=k, text=v) for k, v in zip(STATS, texts)])


def _distributions(run: Run, actual: np.ndarray, pred: np.ndarray,
                   width: float) -> dict[str, Any]:
    a_bin = np.floor(actual / width).astype(int)
    p_bin = np.floor(pred / width).astype(int)
    start = int(min(a_bin.min(), p_bin.min()))
    indices, inverse = np.unique(np.r_[a_bin, p_bin] - start,
                                 return_inverse=True)
    size = int(indices[-1]) + 1
    a_counts = np.bincount(inverse[:len(actual)], minlength=len(indices))
    p_counts = np.bincount(inverse[len(actual):], minlength=len(indices))
    heavy = indices[a_counts + p_counts > FOLD_AT_MOST]
    kept = int(heavy[-1]) + 1 if len(heavy) else size
    folded = kept < size
    if folded:
        keep = indices < kept
        a_counts = np.r_[a_counts[keep], a_counts[~keep].sum()]
        p_counts = np.r_[p_counts[keep], p_counts[~keep].sum()]
        indices = np.r_[indices[keep], kept]
    bin_count = kept + int(folded)
    sparse = {}
    if bin_count <= 1000:
        dense = np.zeros((2, bin_count), dtype=int)
        dense[:, indices] = [a_counts, p_counts]
        a_counts, p_counts = dense
        labels = _edges(width, kept, folded, start)
    else:
        sparse = {'indices': indices.tolist(), 'bin_count': bin_count}
        labels = [_edges(width, 1, False, start + int(i))[0] for i in indices]
        if folded:
            labels[-1] = f'≥ {fmt.number((start + kept) * width)}'
    series = []
    for name, values, counts, word in (('actual', actual, a_counts, 'actual'),
                                       ('predicted', pred, p_counts,
                                        'prediction')):
        stats = _stats(values)
        series.append(
            dict(name=name, counts=counts.tolist(), stats=stats,
                 max_callout=f'highest {word} {stats["items"][-1]["text"]}'))
    return dict(id='distributions', label='Distributions', x=run.target()
                or 'value', y=run.entities, bins=labels, **sparse,
                start=start * width, width=width, series=series)


def predicted_card(run: Run) -> Optional[dict[str, Any]]:
    if run.model_type != REGRESSION:
        return None
    views = []
    width = bin_width(run)
    holdout = run.holdout
    actual = pred = np.empty(0)
    if holdout is not None:
        actual = holdout['TARGET'].to_numpy(dtype=float)
        pred = holdout['TARGET_PRED'].to_numpy(dtype=float)
        keep = np.isfinite(actual) & np.isfinite(pred)
        actual, pred = actual[keep], pred[keep]
        if len(actual):
            entities = holdout['ENTITY'].to_numpy()[keep]
            by_tenth = tenths(run, entities, pred, actual,
                              ('predicted', 'actual'))
            by_tenth.update(label='By predicted tenth',
                            x='predicted tenth, lowest to highest',
                            y='average ' + (run.target(short=True) or 'value'),
                            series=['average prediction', 'average actual'])
            views += [by_tenth, _bands(run, actual, pred, entities)]
    heatmap = _heatmap(run, width)
    if heatmap is not None:
        views.append(heatmap)
    if width is not None and len(actual):
        views.append(_distributions(run, actual, pred, width))
    if not views:
        return None
    return dict(title='Predicted against actual', views=views,
                coverage=run.holdout_coverage())
