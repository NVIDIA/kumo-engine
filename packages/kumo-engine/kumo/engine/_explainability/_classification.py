# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary classification cards: Classification cutoff and Scores against
outcomes, both from the published model's holdout predictions.

The cutoff card carries every count and text for the 101 grid cutoffs of
both positive classes, so the front end only selects what the slider names.
"""

import math
from typing import Any, Optional

import numpy as np

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._run import Run, tenths

GRID = [round(i / 100, 2) for i in range(101)]
DEFAULT_CUTOFF = 50
SCORE_BIN = 0.02


def _ratio(numerator: int, denominator: int, empty: float = 1.0) -> float:
    return numerator / denominator if denominator else empty


def _text(value: float) -> str:
    """A rate in ``[0, 1]`` to three significant figures, zero included."""
    return '0.00' if value == 0 else fmt.number(value)


def counts(scores: np.ndarray, positive: np.ndarray) -> list[list[int]]:
    """``[tp, fp, fn, tn]`` at each grid cutoff, counting ``score > t``."""
    hits, misses = np.sort(scores[positive]), np.sort(scores[~positive])
    tp = len(hits) - np.searchsorted(hits, GRID, side='right')
    fp = len(misses) - np.searchsorted(misses, GRID, side='right')
    return np.stack([tp, fp, len(hits) - tp, len(misses) - fp], axis=1).tolist()


def _at(row: list[int], n: int, positive: bool,
        cutoff: float) -> dict[str, Any]:
    tp, fp, fn, tn = row
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn, empty=0.0)
    flagged = (tp + fp) / n
    fpr = _ratio(fp, fp + tn)
    strip = (('precision', _text(precision)), ('recall', _text(recall)),
             ('accuracy', _text(_ratio(tp + tn,
                                       n))), ('flagged', fmt.percent(flagged)),
             ('F1', _text(_ratio(2 * tp, 2 * tp + fp + fn, empty=0.0))))
    t = f'{cutoff:.2f}'
    return dict(
        counts=row,
        cells=[f'{c:,} · {fmt.percent(c / n)}' for c in row],
        strip=[dict(label=label, text=text) for label, text in strip],
        flagged=flagged,
        precision=precision,
        recall=recall,
        fpr=fpr,
        matrix_callout=f'{fmt.percent(flagged)} flagged at cutoff {t}',
        roc_callout=f'cutoff {t} · TPR {_text(recall)} · FPR {_text(fpr)}',
        pr_callout=(f'cutoff {t} · precision {_text(precision)} · recall '
                    f'{_text(recall)}'),
        gain_callout=(f'{fmt.percent(flagged)} flagged · '
                      f'{fmt.percent(recall)} of {positive} captured'),
        # The score histogram's axis is True_PROB, so its cutoff line is
        # labelled with the value on that axis.
        line_callout=f'cutoff {cutoff if positive else 1 - cutoff:.2f}')


def _curves(scores: np.ndarray, positive: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind='mergesort')
    ends = np.r_[np.where(np.diff(scores[order]))[0], len(scores) - 1]
    tps = np.cumsum(positive[order])[ends].astype(float)
    fps = (ends + 1) - tps
    out = {}
    if tps[-1] > 0:
        recall = tps / tps[-1]
        precision = tps / (tps + fps)
        out['ap'] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        if fps[-1] > 0:
            # Trapezoid area, which counts tied scores half, for runs whose
            # published AUROC is not stored.
            fpr, tpr = np.r_[0.0, fps / fps[-1]], np.r_[0.0, recall]
            out['auroc'] = float(np.sum(
                np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return out


def _class(run: Run, positive: bool, auroc: Optional[float]) -> dict[str, Any]:
    holdout = run.holdout
    assert holdout is not None
    name, other = str(positive), str(not positive)
    target = holdout['TARGET'].astype(bool).to_numpy()
    hits = target if positive else ~target
    share = float(hits.mean())
    rows = counts(holdout['True_PROB'].to_numpy(dtype=float), target)
    if not positive:
        # Match ClassificationEvaluationResponse's reversed class counts.
        rows = [[tn, fn, fp, tp] for tp, fp, fn, tn in reversed(rows)]
    curves = _curves(holdout[f'{name}_PROB'].to_numpy(dtype=float), hits)
    auroc = curves.get('auroc') if auroc is None else auroc
    cutoffs = [_at(row, len(target), positive, t) for row, t in zip(rows, GRID)]
    # UI curves use the same threshold records as the cutoff readout.
    lines = {
        key: [[round(point[x], 2), round(point[y], 2)] for point in cutoffs]
        for key, x, y in (('roc', 'fpr', 'recall'),
                          ('pr', 'recall', 'precision'), ('gain', 'flagged',
                                                          'recall'))
    }
    return dict(
        name=name,
        share=share,
        cutoffs=cutoffs,
        **lines,
        # The grid cutoff whose flagged share is closest to the class share.
        preset=int(np.argmin([abs(c['flagged'] - share) for c in cutoffs])),
        auroc_callout=None if auroc is None else f'AUROC {fmt.number(auroc)}',
        ap_callout=None
        if 'ap' not in curves else f'AP {fmt.number(curves["ap"])}',
        base_rate_callout=f'base rate {fmt.number(share)}',
        ideal_callout=f'ideal bend {fmt.percent(share)}',
        matrix_rows=[f'actual {name}', f'actual {other}'],
        matrix_columns=[f'predicted {name}', f'predicted {other}'])


def cutoff_card(run: Run) -> Optional[dict[str, Any]]:
    if run.task_type != 'binary_classification' or run.holdout is None:
        return None
    published = run.metrics.get('test_refit_auroc',
                                run.metrics.get('test_auroc'))
    views = [
        dict(id='matrix', label='Confusion matrix'),
        dict(id='roc', label='ROC', x='false positive rate',
             y='true positive rate', diagonal='chance'),
        dict(id='pr', label='Precision-recall', x='recall', y='precision'),
        dict(id='gain', label='Cumulative gain',
             x=f'share of {run.entities} flagged',
             y='share of positives captured', diagonal='random'),
    ]
    return dict(
        title='Classification cutoff', grid=GRID,
        grid_text=[f'{t:.2f}' for t in GRID], default=DEFAULT_CUTOFF, classes=[
            _class(run, positive, published) for positive in (True, False)
        ], views=views, coverage=run.holdout_coverage())


def _distribution(run: Run, scores: np.ndarray,
                  target: np.ndarray) -> dict[str, Any]:
    low = math.floor(scores.min() / SCORE_BIN + 1e-9)
    high = max(math.ceil(scores.max() / SCORE_BIN - 1e-9), low + 1)
    edges = np.round(np.arange(low, high + 1) * SCORE_BIN, 2)
    series = []
    for name, mask in (('True', target), ('False', ~target)):
        median = float(np.median(scores[mask])) if mask.any() else None
        series.append(
            dict(
                name=f'actual {name}', counts=np.histogram(scores[mask],
                                                           edges)[0].tolist(),
                median=median, median_callout=None
                if median is None else f'median {fmt.number(median)}'))
    total = np.histogram(scores, edges)[0]
    i = int(np.argmax(total))
    return dict(
        id='outcome', label='Scores by outcome',
        x='predicted probability of True', y=run.entities, edges=edges.tolist(),
        series=series, tallest=i,
        tallest_callout=(f'{total[i]:,} {run.entities} '
                         f'({fmt.percent(total[i] / len(scores))}) score '
                         f'{edges[i]:.2f} to {edges[i + 1]:.2f}'))


def scores_card(run: Run) -> Optional[dict[str, Any]]:
    if run.task_type != 'binary_classification' or run.holdout is None:
        return None
    target = run.holdout['TARGET'].astype(bool).to_numpy()
    scores = run.holdout['True_PROB'].to_numpy(dtype=float)
    by_tenth = tenths(run, run.holdout['ENTITY'].to_numpy(), scores,
                      target.astype(float), ('score', 'observed'))
    by_tenth.update(label='By score tenth', x='score tenth, lowest to highest',
                    y='share True',
                    series=['average score', 'observed share True'])
    return dict(title='Scores against outcomes',
                views=[by_tenth, _distribution(run, scores, target)],
                coverage=run.holdout_coverage())
