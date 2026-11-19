# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cards each model type shows, in order, on its two tabs.

A card whose inputs are missing is left out; a tab with no card is left out.
"""

from collections.abc import Callable
from typing import Any, Optional

from kumo.engine._explainability import (
    _classification,
    _columns,
    _metrics,
    _subgraphs,
)
from kumo.engine._explainability._forecasting import forecast_card
from kumo.engine._explainability._header import header
from kumo.engine._explainability._regression import predicted_card
from kumo.engine._explainability._run import (
    CLASS_LABELED,
    CLASSIFICATION,
    FORECASTING,
    LINK_PREDICTION,
    REGRESSION,
    Run,
)

Card = Callable[[Run], Optional[dict[str, Any]]]

EVALUATION: dict[str, list[tuple[str, Card]]] = {
    CLASSIFICATION: [
        ('metrics', _metrics.card),
        ('cutoff', _classification.cutoff_card),
        ('scores', _classification.scores_card),
    ],
    REGRESSION: [('metrics', _metrics.card), ('predicted', predicted_card)],
    FORECASTING: [('metrics', _metrics.card), ('forecast', forecast_card)],
    LINK_PREDICTION: [('metrics', _metrics.card)],
}

EXPLANATION: dict[str, list[tuple[str, Card]]] = {
    CLASSIFICATION: [('subgraphs', _subgraphs.card),
                     ('importance', _columns.importance),
                     ('groups', _columns.groups)],
    REGRESSION: [('subgraphs', _subgraphs.card),
                 ('importance', _columns.importance),
                 ('groups', _columns.groups)],
    FORECASTING: [('importance', _columns.importance)],
    LINK_PREDICTION: [('subgraphs', _subgraphs.card),
                      ('importance', _columns.importance)],
}


def _cards(run: Run, spec: list[tuple[str, Card]]) -> list[dict[str, Any]]:
    out = []
    for card_id, build in spec:
        if run.task_type in CLASS_LABELED and card_id in ('cutoff', 'scores'):
            continue
        card = build(run)
        if card is not None:
            out.append({'id': card_id, **card})
    return out


def build(run: Run) -> dict[str, Any]:
    """Every number and text the widget draws for one run."""
    tabs = []
    for tab_id, label, spec in (('evaluation', 'Evaluation', EVALUATION),
                                ('explanation', 'Explanation', EXPLANATION)):
        cards = _cards(run, spec[run.model_type])
        if cards:
            tabs.append({'id': tab_id, 'label': label, 'cards': cards})
    return {
        'model_type': run.model_type,
        'task_type': run.task_type,
        'entities': run.entities,
        'header': header(run),
        'tabs': tabs
    }
