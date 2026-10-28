# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notebook plots for retained training explanations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from kumoapi.explain.column_analysis import ColumnCohorts
from kumoapi.explain.gradient import GraphGradientScore

from kumo.engine.explain._artifacts import validate_cohort

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_gradient(
    scores: GraphGradientScore,
    *,
    top_k: int = 20,
) -> tuple[Figure, Axes]:
    """Plot the highest recorded gradient scores, with table-qualified labels.

    Returns a figure and its axis without displaying the figure. Zero scores
    remain zero; scores are never normalized into percentages.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError('top_k must be a positive integer')
    rows = []
    for table_name, table in scores.tables.items():
        for column_name, score in table.columns.items():
            if not table_name or not column_name:
                raise ValueError('Gradient scores require table and column names')
            if not math.isfinite(score):
                raise ValueError('Gradient scores must be finite')
            rows.append((table_name, column_name, score))
    if not rows:
        raise ValueError('No gradient scores are available to plot')
    rows.sort(key=lambda row: (-row[2], row[0], row[1]))
    selected = rows[:top_k]

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name and error.name.split('.')[0] == 'matplotlib':
            raise ImportError(
                'Plotting requires Matplotlib; install kumo-engine[plots].') from error
        raise
    figure, axis = plt.subplots(figsize=(8, max(3, len(selected) * .35)))
    axis.barh([f'{table}.{column}' for table, column, _ in selected],
              [score for _, _, score in selected])
    axis.invert_yaxis()
    axis.set_xlabel('Gradient score')
    axis.set_title('Training column importance')
    figure.tight_layout()
    return figure, axis


def plot_cohort(
    cohort: ColumnCohorts,
    *,
    class_name: str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot cohort predictions and optional targets above population shares.

    Classification outputs with class-keyed predictions require ``class_name``.
    Population shares occupy a separate axis because they have different units
    from predictions and observed targets. Returns without displaying.
    """
    validate_cohort(cohort)
    predictions = cohort.predictions
    targets = cohort.targets
    if isinstance(predictions, dict):
        if class_name is None:
            raise ValueError('class_name is required for class-keyed predictions')
        if class_name not in predictions:
            raise ValueError(f'Unknown class {class_name!r}')
        values = predictions[class_name]
        observed = targets[class_name] if isinstance(targets, dict) else None
    else:
        if class_name is not None:
            raise ValueError('class_name requires class-keyed predictions')
        values, observed = predictions, targets

    count = len(cohort.cohorts)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name and error.name.split('.')[0] == 'matplotlib':
            raise ImportError(
                'Plotting requires Matplotlib; install kumo-engine[plots].') from error
        raise
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                gridspec_kw={'height_ratios': [2, 1]})
    prediction_axis, population_axis = axes
    positions = list(range(count))
    prediction_axis.plot(positions, values, marker='o', label='Prediction')
    if observed is not None:
        prediction_axis.plot(positions, observed, marker='o', label='Observed target')
    prediction_axis.set_ylabel('Average value')
    title = f'{cohort.table_name}[hop={cohort.hop}].{cohort.col_name}'
    prediction_axis.set_title(f'{title} — {class_name}' if class_name else title)
    prediction_axis.legend()
    population_axis.bar(positions, cohort.populations)
    population_axis.set_ylim(0, 1)
    population_axis.set_ylabel('Population share')
    population_axis.set_xticks(positions, cohort.cohorts, rotation=35, ha='right')
    population_axis.set_xlabel('Cohort')
    figure.tight_layout()
    return figure, (prediction_axis, population_axis)
