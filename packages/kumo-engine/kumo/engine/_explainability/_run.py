# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One training run's inputs, read once and shared by every card."""

import math
import posixpath
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from kumo.engine._explainability import _files
from kumo.engine._explainability._coverage import Coverage, detect

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
FORECASTING = 'forecasting'
LINK_PREDICTION = 'link_prediction'

CLASS_LABELED = ('multiclass_classification', 'multilabel_classification')

# Kumo task types the widget supports, by the model type it renders.
_TASKS = {
    'binary_classification': CLASSIFICATION,
    **dict.fromkeys(CLASS_LABELED, CLASSIFICATION),
    'regression': REGRESSION,
    'forecasting': FORECASTING,
    'link_prediction': LINK_PREDICTION,
    'temporal_link_prediction': LINK_PREDICTION,
    'static_link_prediction': LINK_PREDICTION,
}

_HOLDOUT_COLUMNS = {
    CLASSIFICATION: ['ENTITY', 'TARGET', 'True_PROB', 'False_PROB'],
    REGRESSION: ['ENTITY', 'TARGET', 'TARGET_PRED'],
    FORECASTING:
    ['ENTITY', 'TIMESTAMP', 'ANCHOR_TIMESTAMP', 'TARGET', 'TARGET_PRED'],
}


class UnsupportedModelError(ValueError):
    """The run's task type has no explainability widget."""


@dataclass
class Run:
    files: _files.RunFiles
    model_type: str
    plan: dict[str, Any]
    metrics: dict[str, float]
    entities: str
    holdout: Optional[pd.DataFrame]
    coverage: Coverage
    task_type: str = ''

    @property
    def training_job(self) -> dict[str, Any]:
        job = self.plan.get('training_job')
        return job if isinstance(job, dict) else {}


def model_type(task_type: Optional[str]) -> str:
    if task_type not in _TASKS:
        raise UnsupportedModelError(
            f'No explainability view for task type {task_type!r}; supported: '
            'binary, multiclass and multilabel classification, regression, forecasting and link '
            'prediction.')
    return _TASKS[task_type]


def load(files: _files.RunFiles) -> Run:
    task = _files.task_type(files)
    kind = model_type(task)
    metrics = _files.read_json(posixpath.join(files.bundle, 'metrics.json'))
    metrics = metrics if isinstance(metrics, dict) else {}
    metrics = {
        key: float(value)
        for key, value in metrics.items() if isinstance(value, (int, float))
        and not isinstance(value, bool) and math.isfinite(value)
    }
    columns = ['ENTITY', 'TARGET'
               ] if task in CLASS_LABELED else _HOLDOUT_COLUMNS.get(kind)
    holdout = None if columns is None else _files.holdout(files, columns)
    entities = _files.entity_table(files) or 'rows'
    return Run(files, kind, _files.plan(files), metrics, entities, holdout,
               detect(files, holdout), task_type=task or '')
