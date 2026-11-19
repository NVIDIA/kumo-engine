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
from kumo.engine._explainability._holdout import (
    equal_count_groups,
    finite,
)
from kumo.engine._explainability._run import Run

GRID = [round(i / 100, 2) for i in range(101)]
DEFAULT_CUTOFF = 50
SCORE_BIN = 0.02
CLASSES = ('True', 'False')


def _ratio(numerator: int, denominator: int, empty: float = 1.0) -> float:
    return numerator / denominator if denominator else empty


def _text(value: float) -> str:
    """A rate in ``[0, 1]`` to three significant figures, zero included."""
    if not math.isfinite(value):
        return 'N/A'
    return '0.00' if value == 0 else fmt.number(value)


def counts(scores: np.ndarray, positive: np.ndarray) -> list[list[int]]:
    """``[tp, fp, fn, tn]`` at each grid cutoff, counting ``score > t``."""
    out = []
    for t in GRID:
        flagged = scores > t
        out.append([
            int((flagged & positive).sum()),
            int((flagged & ~positive).sum()),
            int((~flagged & positive).sum()),
            int((~flagged & ~positive).sum()),
        ])
    return out


def preset(rows: list[list[int]], share: float) -> int:
    """Grid index whose flagged share is closest to the positive share."""
    n = sum(rows[0])
    gaps = [abs((tp + fp) / n - share) for tp, fp, _, _ in rows]
    return gaps.index(min(gaps))


def _curves(scores: np.ndarray, positive: np.ndarray) -> dict[str, Any]:
    n = len(scores)
    order = np.argsort(-scores, kind='mergesort')
    ranked, hits = scores[order], positive[order]
    ends = np.r_[np.where(np.diff(ranked))[0], n - 1]
    tps = np.cumsum(hits)[ends].astype(float)
    fps = (ends + 1) - tps
    n_pos, n_neg = tps[-1], fps[-1]
    out: dict[str, Any] = {}
    if n_pos > 0:
        recall = tps / n_pos
        precision = tps / (tps + fps)
        ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        out['ap'] = ap
        if n_neg > 0:
            # Trapezoid area, which counts tied scores half, for runs whose
            # published AUROC is not stored.
            fpr = np.r_[0.0, fps / n_neg]
            tpr = np.r_[0.0, recall]
            out['auroc'] = float(np.sum(
                np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return out


def _at(row: list[int], n: int, name: str, cutoff: float) -> dict[str, Any]:
    tp, fp, fn, tn = row
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn, empty=0.0)
    accuracy = _ratio(tp + tn, n)
    flagged = (tp + fp) / n
    fpr = _ratio(fp, fp + tn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn, empty=0.0)
    t = fmt.cutoff(cutoff)
    # The score histogram's axis is the first class's probability, so its
    # cutoff line is labelled with the value on that axis (W5).
    on_axis = fmt.cutoff(cutoff if name == CLASSES[0] else 1 - cutoff)
    recall_share = fmt.percent(recall) if math.isfinite(recall) else 'N/A'
    return {
        'counts':
        row,
        'cells': [f'{fmt.count(c)} · {fmt.percent(c / n)}' for c in row],
        'strip': [
            {
                'label': 'precision',
                'text': _text(precision)
            },
            {
                'label': 'recall',
                'text': _text(recall)
            },
            {
                'label': 'accuracy',
                'text': _text(accuracy)
            },
            {
                'label': 'flagged',
                'text': fmt.percent(flagged)
            },
            {
                'label': 'F1',
                'text': _text(f1)
            },
        ],
        'flagged':
        flagged,
        'precision':
        finite(precision),
        'recall':
        finite(recall),
        'fpr':
        finite(fpr),
        'matrix_callout':
        f'{fmt.percent(flagged)} flagged at cutoff {t}',
        'roc_callout':
        f'cutoff {t} · TPR {_text(recall)} · FPR {_text(fpr)}',
        'pr_callout':
        f'cutoff {t} · precision {_text(precision)} · recall {_text(recall)}',
        'gain_callout':
        f'{fmt.percent(flagged)} flagged · {recall_share} of {name} captured',
        'line_callout':
        f'cutoff {on_axis}',
    }


def _class(run: Run, name: str, auroc: Optional[float]) -> dict[str, Any]:
    holdout = run.holdout
    assert holdout is not None
    target = holdout['TARGET'].astype(bool).to_numpy()
    positive = target if name == 'True' else ~target
    scores = holdout[f'{name}_PROB'].to_numpy(dtype=float)
    n = len(scores)
    share = float(positive.mean())
    rows = counts(holdout['True_PROB'].to_numpy(dtype=float), target)
    if name == 'False':
        # Match ClassificationEvaluationResponse's reversed class counts.
        rows = [[tn, fn, fp, tp] for tp, fp, fn, tn in reversed(rows)]
    curves = _curves(scores, positive)
    if auroc is None:
        auroc = curves.get('auroc')
    other = 'False' if name == 'True' else 'True'
    cutoffs = [_at(row, n, name, t) for row, t in zip(rows, GRID)]
    # UI curves use the same threshold records as the cutoff readout.
    for key, x, y in (('roc', 'fpr', 'recall'), ('pr', 'recall', 'precision'),
                      ('gain', 'flagged', 'recall')):
        curves[key] = [[round(point[x], 2),
                        round(point[y], 2)] for point in cutoffs]
    return {
        'name': name,
        'other': other,
        'share': share,
        'preset': preset(rows, share),
        'cutoffs': cutoffs,
        'roc': curves.get('roc'),
        'pr': curves.get('pr'),
        'gain': curves.get('gain'),
        'auroc': auroc,
        'auroc_callout':
        None if auroc is None else f'AUROC {fmt.number(auroc)}',
        'ap': curves.get('ap'),
        'ap_callout':
        None if 'ap' not in curves else f'AP {fmt.number(curves["ap"])}',
        'base_rate_callout': f'base rate {fmt.number(share)}',
        'ideal_callout': f'ideal bend {fmt.percent(share)}',
        'matrix_rows': [f'actual {name}', f'actual {other}'],
        'matrix_columns': [f'predicted {name}', f'predicted {other}'],
    }


def cutoff_card(run: Run) -> Optional[dict[str, Any]]:
    holdout = run.holdout
    if holdout is None or len(holdout) == 0:
        return None
    published = run.metrics.get('test_refit_auroc',
                                run.metrics.get('test_auroc'))
    classes = [_class(run, name, published) for name in CLASSES]
    return {
        'title':
        'Classification cutoff',
        'grid':
        GRID,
        'grid_text': [fmt.cutoff(t) for t in GRID],
        'default':
        DEFAULT_CUTOFF,
        'classes':
        classes,
        'views': [
            {
                'id': 'matrix',
                'label': 'Confusion matrix'
            },
            {
                'id': 'roc',
                'label': 'ROC',
                'x': 'false positive rate',
                'y': 'true positive rate',
                'diagonal': 'chance'
            },
            {
                'id': 'pr',
                'label': 'Precision-recall',
                'x': 'recall',
                'y': 'precision'
            },
            {
                'id': 'gain',
                'label': 'Cumulative gain',
                'x': f'share of {run.entities} flagged',
                'y': 'share of positives captured',
                'diagonal': 'random'
            },
        ],
        'coverage':
        run.coverage.holdout_line(),
    }


def _tenths(run: Run, scores: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    holdout = run.holdout
    assert holdout is not None
    group = equal_count_groups(scores, holdout['ENTITY'].to_numpy())
    occupied = np.unique(group)
    rows = []
    for k in occupied:
        mask = group == k
        rows.append((int(mask.sum()), float(scores[mask].mean()),
                     float(target[mask].mean())))
    gaps = [abs(p - o) for _, p, o in rows]
    worst = gaps.index(max(gaps))
    share = float(target.mean())
    _, p, o = rows[worst]
    return {
        'id':
        'tenths',
        'label':
        'By score tenth',
        'x':
        'score tenth, lowest to highest',
        'y':
        'share True',
        'series': ['average score', 'observed share True'],
        'tenths': [{
            'label': str(occupied[k] + 1),
            'rows': n,
            'score': p_,
            'score_text': fmt.number(p_),
            'observed': o_,
            'observed_text': fmt.number(o_),
        } for k, (n, p_, o_) in enumerate(rows)],
        'reference':
        share,
        'reference_callout':
        f'all {run.entities}: {fmt.number(share)}',
        'highlight':
        worst,
        'gap_callout':
        f'tenth {occupied[worst] + 1}: predicted {fmt.number(p)} · observed '
        f'{fmt.number(o)}',
    }


def _distribution(run: Run, scores: np.ndarray,
                  target: np.ndarray) -> dict[str, Any]:
    low = math.floor(scores.min() / SCORE_BIN + 1e-9)
    high = math.ceil(scores.max() / SCORE_BIN - 1e-9)
    high = max(high, low + 1)
    edges = np.round(np.arange(low, high + 1) * SCORE_BIN, 2)
    true_counts, _ = np.histogram(scores[target], edges)
    false_counts, _ = np.histogram(scores[~target], edges)
    total = true_counts + false_counts
    tallest = int(np.argmax(total))
    n = len(scores)
    medians = {
        'True': float(np.median(scores[target])) if target.any() else None,
        'False': float(np.median(scores[~target])) if (~target).any() else None,
    }
    return {
        'id':
        'outcome',
        'label':
        'Scores by outcome',
        'x':
        'predicted probability of True',
        'y':
        run.entities,
        'edges': [float(e) for e in edges],
        'series': [
            {
                'name':
                'actual True',
                'counts': [int(c) for c in true_counts],
                'median':
                medians['True'],
                'median_callout': (f'median {fmt.number(medians["True"])}'
                                   if medians['True'] is not None else None),
            },
            {
                'name':
                'actual False',
                'counts': [int(c) for c in false_counts],
                'median':
                medians['False'],
                'median_callout': (f'median {fmt.number(medians["False"])}'
                                   if medians['False'] is not None else None),
            },
        ],
        'tallest':
        tallest,
        'tallest_callout':
        (f'{fmt.count(total[tallest])} {run.entities} '
         f'({fmt.percent(total[tallest] / n)}) score '
         f'{fmt.cutoff(edges[tallest])} to {fmt.cutoff(edges[tallest + 1])}'),
    }


def scores_card(run: Run) -> Optional[dict[str, Any]]:
    holdout = run.holdout
    if holdout is None or len(holdout) == 0:
        return None
    target = holdout['TARGET'].astype(bool).to_numpy()
    scores = holdout['True_PROB'].to_numpy(dtype=float)
    return {
        'title': 'Scores against outcomes',
        'views':
        [_tenths(run, scores, target),
         _distribution(run, scores, target)],
        'coverage': run.coverage.holdout_line(),
    }
