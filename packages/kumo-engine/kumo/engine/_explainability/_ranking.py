# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ranking quality by cutoff K, for multilabel ranking and link prediction.

Every value is the one ``TrainingResult.metrics`` reports: the evaluated
fit's holdout scores (before full refit when applicable), and the original
run's validation and last-epoch training scores.
"""

from typing import Any, Optional

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._metrics import (
    _family,
    _holdout,
    _prefixed,
    _top_k,
    display_name,
)
from kumo.engine._explainability._run import LINK_PREDICTION, Run

_FAMILIES = ('recall', 'precision', 'map', 'ndcg', 'f1')


def _splits(run: Run) -> list[tuple[str, str, dict[str, float]]]:
    return [('holdout',
             'Holdout (before full refit)' if run.full_refit else 'Holdout',
             _holdout(run)), ('val', 'Validation', _prefixed(run, 'val_')),
            ('train', 'Training', _prefixed(run, 'train_'))]


def card(run: Run) -> Optional[dict[str, Any]]:
    if run.model_type != LINK_PREDICTION:
        return None
    splits = _splits(run)
    by_family: dict[str, dict[int, str]] = {}
    for metric in run.metric_names:
        k = _top_k(metric)
        if k is not None:
            names = by_family.setdefault(_family(metric), {})
            if k not in names or metric == f'{_family(metric)}@{k}':
                names[k] = metric
    tuned = _family(run.tune or '') if _top_k(run.tune
                                              or '') is not None else None
    order = sorted(
        by_family, key=lambda family:
        (family != tuned, _FAMILIES.index(family)
         if family in _FAMILIES else len(_FAMILIES), family))
    views = []
    for family in order:
        ks = sorted(by_family[family])
        series = []
        for split, label, values in splits:
            found = [values.get(by_family[family][k]) for k in ks]
            if any(v is not None for v in found):
                series.append(
                    dict(
                        id=split, label=label, values=found, texts=[
                            None if v is None else fmt.number(v) for v in found
                        ]))
        if len(ks) > 1 and series:
            name = display_name(f'{family}@{ks[0]}').split('@')[0]
            views.append(
                dict(id=family, label=name, ks=ks, series=series,
                     x='K (number of items ranked)', y=name))
    if not views:
        return None
    return dict(title='Ranking quality by cutoff K', views=views,
                coverage=run.metrics_coverage())
