# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ranking quality by cutoff K, for multilabel ranking and link prediction.

Every value is the one ``TrainingResult.metrics`` reports: the published
model's holdout scores, and the original run's validation and last-epoch
training scores.
"""

import re
from typing import Any, Optional

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._metrics import (
    _holdout,
    _prefixed,
    display_name,
)
from kumo.engine._explainability._run import LINK_PREDICTION, Run

_FAMILIES = ('recall', 'precision', 'map', 'ndcg', 'f1')
_AT_K = re.compile(r'([a-z0-9_]+)@(\d+)')


def _splits(run: Run) -> list[tuple[str, str, dict[str, float]]]:
    return [('holdout',
             'Holdout (before full refit)' if run.full_refit else 'Holdout',
             _holdout(run)), ('val', 'Validation', _prefixed(run, 'val_')),
            ('train', 'Training', _prefixed(run, 'train_'))]


def card(run: Run) -> Optional[dict[str, Any]]:
    if run.model_type != LINK_PREDICTION:
        return None
    splits = _splits(run)
    by_family: dict[str, set[int]] = {}
    for metric in run.metric_names:
        match = _AT_K.fullmatch(metric)
        if match:
            by_family.setdefault(match[1], set()).add(int(match[2]))
    tuned = _AT_K.fullmatch(run.tune or '')
    order = sorted(
        by_family, key=lambda family:
        (family != (tuned and tuned[1]), _FAMILIES.index(family)
         if family in _FAMILIES else len(_FAMILIES), family))
    views = []
    for family in order:
        ks = sorted(by_family[family])
        series = []
        for split, label, values in splits:
            found = [values.get(f'{family}@{k}') for k in ks]
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
                     x='K (number of items ranked)', y=name,
                     tune=bool(tuned and tuned[1] == family)))
    if not views:
        return None
    return dict(title='Ranking quality by cutoff K', views=views,
                coverage=run.metrics_coverage())
