# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Metrics card, shared by every model type, and the widget header.

Holdout scores describe the evaluated fit, before any full-data refit.
Training and validation scores use the original run's last epoch.
"""

import math
import re
from typing import Any, Optional

import numpy as np

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._run import (
    CLASS_LABELED,
    FORECASTING,
    LINK_PREDICTION,
    REGRESSION,
    Run,
)

_NAMES = dict(auroc='AUROC', ap='AP', auprc='AUPRC', mae='MAE', mse='MSE',
              rmse='RMSE', loss='Loss', map='MAP', recall='Recall',
              precision='Precision', f1='F1', acc='Accuracy',
              accuracy='Accuracy', smape='SMAPE', r2='R²', mape='MAPE',
              ndcg='NDCG')
_RANKING_ORDER = ('f1', 'map', 'ndcg', 'precision', 'recall')

Baselines = dict[str, list[tuple[str, float]]]


def _top_k(metric: str) -> Optional[int]:
    match = re.fullmatch(r'[a-z0-9_]+@(\d+)', metric)
    return int(match.group(1)) if match else None


def display_name(metric: str) -> str:
    family = metric.split('@')[0]
    base, _, average = family.partition('_')
    if average in ('macro', 'micro', 'weighted') and base in _NAMES:
        return f'{_NAMES[base]} ({average})'
    name = _NAMES.get(family, family.upper())
    k = _top_k(metric)
    return name if k is None else f'{name}@{k}'


def _prefixed(run: Run, prefix: str) -> dict[str, float]:
    return {
        key.removeprefix(prefix): value
        for key, value in run.metrics.items() if key.startswith(prefix)
    }


def _refit(run: Run) -> bool:
    return run.refit or any(
        key.startswith('test_refit_') for key in run.metrics)


def _holdout(run: Run) -> dict[str, float]:
    if _refit(run):
        return _prefixed(run, 'test_refit_')
    return {
        key: value
        for key, value in _prefixed(run, 'test_').items()
        if not key.startswith('refit_')
    }


def baselines(run: Run) -> Baselines:
    out: Baselines = {}
    if run.task_type in CLASS_LABELED:
        return out
    if run.model_type == LINK_PREDICTION and run.catalog is not None:
        size, mean_targets, _ = run.catalog
        for metric in run.metric_names:
            k = _top_k(metric)
            if k is not None and metric.startswith('recall@'):
                out[metric] = [('random', k / size)]
            elif k is not None and metric.startswith('precision@'):
                out[metric] = [('random', mean_targets / size)]
    elif run.model_type in (REGRESSION,
                            FORECASTING) and run.holdout is not None:
        actual = run.holdout['TARGET'].to_numpy(dtype=float)
        for name, pred in (('predict zero', 0.0), ('predict the mean',
                                                   actual.mean())):
            error = pred - actual
            mse = float((error**2).mean())
            for metric, value in (('mae', float(np.abs(error).mean())),
                                  ('mse', mse), ('rmse', math.sqrt(mse))):
                out.setdefault(metric, []).append((name, value))
    return out


def _family(metric: str) -> str:
    return metric.split('@')[0]


def _position(order: tuple[str, ...], metric: str) -> int:
    family = _family(metric)
    return order.index(family) if family in order else len(order)


def _same_k(metrics: list[str], tune: str, k: int) -> list[str]:
    """Metrics at the tuning metric's K, recall and precision first."""
    rest = [m for m in metrics if _top_k(m) == k and m != tune]
    return sorted(rest, key=lambda m: _position(('recall', 'precision'), m))


def tile_metrics(run: Run) -> list[str]:
    """The main metrics, shown as pills."""
    metrics = run.metric_names
    tune = run.tune or (metrics[0] if metrics else None)
    if tune is None:
        return []
    k = _top_k(tune)
    if k is not None:
        order = [tune] + _same_k(metrics, tune, k)
    else:
        order = [tune] + [
            m for m in metrics if not (m == 'auprc' and 'ap' in metrics)
            and not (m == 'mse' and 'rmse' in metrics)
        ]
    return list(dict.fromkeys(order))[:3]


def _ordered(metrics: list[str]) -> list[str]:
    """Top-K metrics group by family, then K; others keep their order."""
    if not any(_top_k(m) is not None for m in metrics):
        return metrics
    return sorted(metrics, key=lambda m:
                  (_position(_RANKING_ORDER, m), _top_k(m) or 0))


def _table(run: Run, holdout: dict[str, float],
           base: Baselines) -> dict[str, Any]:
    columns = [('Holdout', holdout), ('Validation', _prefixed(run, 'val_')),
               ('Training', _prefixed(run, 'train_'))]
    rows = [(display_name(m), m == run.tune, [v.get(m) for _, v in columns])
            for m in _ordered(list(run.metric_names))]
    if run.model_type == LINK_PREDICTION:
        prefix = 'test_refit_' if _refit(run) else 'test_'
        rows.append(('Locality score', False,
                     [run.metrics.get(prefix + 'locality_score'), None, None]))
    rows = [row for row in rows if any(v is not None for v in row[2])]
    # Drop columns no row fills, such as training values a bundle lacks.
    keep = [i for i in range(3) if any(row[2][i] is not None for row in rows)]
    out = [
        dict(
            name=name, tune=tune, baseline=False, texts=[
                None if values[i] is None else fmt.number(values[i])
                for i in keep
            ]) for name, tune, values in rows
    ]
    # A baseline is scored on the holdout alone; its rows group by baseline.
    named: dict[str, list[tuple[str, float]]] = {}
    for metric, items in base.items():
        for name, value in items:
            named.setdefault(name, []).append((display_name(metric), value))
    for name, items in named.items():
        for metric, value in items:
            out.append(
                dict(
                    name=f'{name[0].upper()}{name[1:]} · {metric}', tune=False,
                    baseline=True,
                    texts=[fmt.number(value) if i == 0 else None
                           for i in keep]))
    return dict(columns=[dict(label=columns[i][0]) for i in keep], rows=out)


def card(run: Run) -> Optional[dict[str, Any]]:
    holdout = _holdout(run)
    shown = [metric for metric in tile_metrics(run) if metric in holdout]
    if not shown:
        return None
    # Baselines are listed for the pill metrics only, to keep the table light.
    base = {
        metric: items
        for metric, items in baselines(run).items() if metric in shown
    }
    # A random baseline needs no rows; the rest come from the retained
    # holdout, which may hold fewer rows than the scored split.
    line = run.holdout_coverage() if any(name != 'random'
                                         for items in base.values()
                                         for name, _ in items) else None
    if line:
        line = 'Baselines: ' + line[0].lower() + line[1:]
    pills = [
        dict(name=display_name(metric), tune=metric == run.tune,
             holdout_text=fmt.number(holdout[metric])) for metric in shown
    ]
    view = dict(id='metrics', label='Metrics', pills=pills,
                table=_table(run, holdout, base), coverage=line)
    return dict(title='Metrics', views=[view], coverage=run.metrics_coverage())


def header(run: Run) -> dict[str, Any]:
    """The query, and a note when the model was refit on all the data."""
    out: dict[str, Any] = dict(query=run.query)
    if run.full_refit:
        out['note'] = ('Diagnostics and explanations describe the model '
                       'before full refit.')
    return out
