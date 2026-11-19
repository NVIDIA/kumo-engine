# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression card: Predicted against actual, from the published model's
holdout predictions and the training table's label histogram bin width."""

import math
from dataclasses import fields
from typing import Any, Optional

import numpy as np

from kumo.engine._explainability import _files
from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._header import target
from kumo.engine._explainability._holdout import FOLD_AT_MOST, equal_count_groups
from kumo.engine._explainability._run import Run

ZERO_BAND_SHARE = 0.10
BANDS = 5
STATS = (('min', 'min'), ('p25', '25th percentile'), ('median', 'median'),
         ('mean', 'mean'), ('p75', '75th percentile'), ('max', 'max'))


def _tenths(run: Run, actual: np.ndarray, pred: np.ndarray,
            entities: np.ndarray) -> dict[str, Any]:
    group = equal_count_groups(pred, entities)
    occupied = np.unique(group)
    means = [(float(pred[group == k].mean()), float(actual[group == k].mean()))
             for k in occupied]
    reference = float(actual.mean())
    texts = fmt.numbers([v for pair in means for v in pair] + [reference])
    gaps = [abs(p - a) for p, a in means]
    worst = gaps.index(max(gaps))
    return {
        'id':
        'tenths',
        'label':
        'By predicted tenth',
        'x':
        'predicted tenth, lowest to highest',
        'y':
        'average ' + (target(run, short=True) or 'value'),
        'series': ['average prediction', 'average actual'],
        'tenths': [{
            'label': str(occupied[k] + 1),
            'predicted': p,
            'predicted_text': texts[2 * k],
            'actual': a,
            'actual_text': texts[2 * k + 1],
        } for k, (p, a) in enumerate(means)],
        'reference':
        reference,
        'reference_callout':
        f'all {run.entities}: {texts[-1]}',
        'highlight':
        worst,
        'gap_callout':
        f'tenth {occupied[worst] + 1}: predicted {texts[2 * worst]} · actual '
        f'{texts[2 * worst + 1]}',
    }


def _bands(run: Run, actual: np.ndarray, pred: np.ndarray,
           entities: np.ndarray) -> dict[str, Any]:
    n = len(actual)
    band = np.full(n, -1)
    zero = actual == 0
    labels: list[str] = []
    if zero.mean() >= ZERO_BAND_SHARE:
        band[zero] = 0
        labels.append('0')
        rest, parts, first = np.where(~zero)[0], BANDS - 1, 1
    else:
        rest, parts, first = np.arange(n), BANDS, 0
    if len(rest):
        group = equal_count_groups(actual[rest], entities[rest], parts)
        band[rest] = first + group
    error = pred - actual
    squared = error**2
    total = float(squared.sum())
    rows = []
    for b in range(first + parts if len(rest) else first):
        mask = band == b
        if not mask.any():
            continue
        if b >= first:
            low, high = fmt.numbers([actual[mask].min(), actual[mask].max()])
            label = f'{low} to {high}'
        else:
            label = labels[0]
        share = float(squared[mask].sum()) / total if total else 0.0
        median = float(np.median(error[mask]))
        rows.append({
            'label': label,
            'rows': int(mask.sum()),
            'rows_text': f'{fmt.count(mask.sum())} {run.entities}',
            'median': median,
            'median_text': fmt.signed(median),
            'error_share': share,
            'error_share_text': fmt.percent(share),
        })
    worst = max(range(len(rows)), key=lambda i: rows[i]['error_share'])
    return {
        'id':
        'bands',
        'label':
        'By actual value',
        'x':
        'actual value',
        'y':
        'median of predicted minus actual',
        'bands':
        rows,
        'highlight':
        worst,
        'share_callout':
        f'{fmt.percent(rows[worst]["error_share"])} of squared error',
    }


def bin_width(run: Run) -> Optional[float]:
    stats = _files.split_stats(run.files) or {}
    test = stats.get('test_table_stats') or {}
    distribution = test.get('target_distribution')
    histogram = distribution.get('histogram') if isinstance(distribution,
                                                            dict) else None
    width = histogram.get('bin_width') if isinstance(histogram, dict) else None
    if isinstance(width, (int, float)) and not isinstance(
            width, bool) and math.isfinite(width) and width > 0:
        return float(width)
    return None


def _bin(values: np.ndarray, width: float) -> np.ndarray:
    return np.floor(values / width).astype(int)


def _edges(width: float, kept: int, folded: bool, start: int = 0) -> list[str]:
    edges = fmt.numbers([(start + i) * width for i in range(kept + 1)])
    labels = [f'{edges[i]} to {edges[i + 1]}' for i in range(kept)]
    if folded:
        labels.append(f'≥ {edges[kept]}')
    return labels


def _heatmap(run: Run, width: Optional[float]) -> Optional[dict[str, Any]]:
    from kumo_core.train.output import EvalOutput, TargetPredictionHeatMap

    saved = _files.read_json(
        _files._join(run.files.trial, 'test_target_prediction_heatmap',
                     'heatmap.json'))
    heatmap = None
    if isinstance(saved, dict):
        try:
            heatmap = TargetPredictionHeatMap(**{
                f.name: saved[f.name]
                for f in fields(TargetPredictionHeatMap)
            })
            bounds = (heatmap.cell_width, heatmap.label_bin_start,
                      heatmap.label_bin_end, heatmap.prediction_bin_start,
                      heatmap.prediction_bin_end)
            if (not all(math.isfinite(v)
                        for v in bounds) or heatmap.cell_width <= 0
                    or heatmap.label_bin_end < heatmap.label_bin_start
                    or heatmap.prediction_bin_end < heatmap.prediction_bin_start
                    or not heatmap.counts
                    or len(heatmap.cells) != len(heatmap.counts)
                    or any(not isinstance(n, int) or n < 0
                           for n in heatmap.counts) or sum(heatmap.counts) == 0
                    or any(not (heatmap.label_bin_start <= x <= heatmap.
                                label_bin_end and heatmap.prediction_bin_start
                                <= y <= heatmap.prediction_bin_end)
                           for x, y in heatmap.cells)):
                heatmap = None
        except (KeyError, TypeError, ValueError):
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
        coverage = (f'Based on {fmt.count(len(sample))} of '
                    f'{fmt.count(len(frame))} retained holdout rows.')
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
    xs = start + np.arange(columns) * cell_width
    ys = start + np.arange(rows) * cell_width
    cells = np.zeros((rows, columns), dtype=int)
    for (x, y), count in zip(heatmap.cells, heatmap.counts):
        row = min(len(ys) - 1, max(0, round((y - ys[0]) / cell_width)))
        col = min(len(xs) - 1, max(0, round((x - xs[0]) / cell_width)))
        cells[row, col] += count

    def view(matrix: np.ndarray) -> dict[str, Any]:
        rows, columns = matrix.shape
        x_edges = [float(xs[0] + i * cell_width) for i in range(columns + 1)]
        y_edges = [float(ys[0] + i * cell_width) for i in range(rows + 1)]
        row, col = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        total = sum(heatmap.counts)
        return {
            'columns': [
                f'{fmt.number(a)} to {fmt.number(b)}'
                for a, b in zip(x_edges, x_edges[1:])
            ],
            'rows': [
                f'{fmt.number(a)} to {fmt.number(b)}'
                for a, b in zip(y_edges, y_edges[1:])
            ],
            'x_edges':
            x_edges,
            'y_edges':
            y_edges,
            'width':
            cell_width,
            'counts':
            matrix.tolist(),
            'densest': [int(row), int(col)],
            'densest_callout':
            f'{fmt.count(matrix[row, col])} {run.entities} '
            f'({fmt.percent(matrix[row, col] / total)})',
        }

    def kept(counts: np.ndarray) -> int:
        indices = np.flatnonzero(counts > 10)
        return int(indices[-1] + 1) if len(indices) else len(counts)

    trimmed = cells[:kept(cells.sum(axis=1)), :kept(cells.sum(axis=0))]
    return {
        'id': 'heatmap',
        'label': 'Heatmap',
        'x': 'actual value',
        'y': 'predicted value',
        **view(cells),
        'trimmed': view(trimmed),
        'coverage': coverage,
        'diagonal': 'predicted = actual',
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    numbers = {
        'min': float(values.min()),
        'p25': float(np.percentile(values, 25)),
        'median': float(np.median(values)),
        'mean': float(values.mean()),
        'p75': float(np.percentile(values, 75)),
        'max': float(values.max()),
    }
    texts = fmt.numbers([numbers[key] for key, _ in STATS])
    return {
        'values':
        numbers,
        'items': [{
            'label': label,
            'text': text
        } for (_, label), text in zip(STATS, texts)],
    }


def _distributions(run: Run, actual: np.ndarray, pred: np.ndarray,
                   width: float) -> dict[str, Any]:
    a_bin, p_bin = _bin(actual, width), _bin(pred, width)
    start = int(min(a_bin.min(), p_bin.min()))
    a_bin, p_bin = a_bin - start, p_bin - start
    indices, inverse = np.unique(np.r_[a_bin, p_bin], return_inverse=True)
    size = int(indices[-1]) + 1
    a_counts = np.bincount(inverse[:len(actual)], minlength=len(indices))
    p_counts = np.bincount(inverse[len(actual):], minlength=len(indices))
    heavy = indices[a_counts + p_counts > FOLD_AT_MOST]
    kept = max(1, int(heavy[-1]) + 1) if len(heavy) else 1
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
    a_stats, p_stats = _stats(actual), _stats(pred)
    return {
        'id':
        'distributions',
        'label':
        'Distributions',
        'x':
        target(run) or 'value',
        'y':
        run.entities,
        'bins':
        labels,
        **sparse,
        'start':
        start * width,
        'width':
        width,
        'series': [
            {
                'name': 'actual',
                'counts': [int(c) for c in a_counts],
                'stats': a_stats,
                'max_callout': f'highest actual {a_stats["items"][-1]["text"]}',
            },
            {
                'name': 'predicted',
                'counts': [int(c) for c in p_counts],
                'stats': p_stats,
                'max_callout':
                f'highest prediction {p_stats["items"][-1]["text"]}',
            },
        ],
    }


def predicted_card(run: Run) -> Optional[dict[str, Any]]:
    views = []
    width = bin_width(run)
    holdout = run.holdout
    actual = pred = None
    if holdout is not None and len(holdout):
        actual = holdout['TARGET'].to_numpy(dtype=float)
        pred = holdout['TARGET_PRED'].to_numpy(dtype=float)
        keep = np.isfinite(actual) & np.isfinite(pred)
        actual, pred = actual[keep], pred[keep]
        if len(actual):
            entities = holdout['ENTITY'].to_numpy()[keep]
            views.extend([
                _tenths(run, actual, pred, entities),
                _bands(run, actual, pred, entities)
            ])
    heatmap = _heatmap(run, width)
    if heatmap is not None:
        views.append(heatmap)
    if width is not None and actual is not None and pred is not None and len(
            actual):
        views.append(_distributions(run, actual, pred, width))
    if not views:
        return None
    return {
        'title': 'Predicted against actual',
        'views': views,
        'coverage': run.coverage.holdout_line(),
    }
