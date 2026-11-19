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
from kumo.engine._explainability._forecasting import period
from kumo.engine._explainability._run import (
    CLASS_LABELED,
    CLASSIFICATION,
    FORECASTING,
    LINK_PREDICTION,
    REGRESSION,
    Run,
    mapping,
)

_NAMES = dict(auroc='AUROC', ap='AP', auprc='AUPRC', mae='MAE', mse='MSE',
              rmse='RMSE', loss='Loss', map='MAP', recall='Recall',
              precision='Precision', f1='F1', acc='Accuracy',
              accuracy='Accuracy', smape='SMAPE', r2='R²', mape='MAPE',
              ndcg='NDCG')
_RANKING_ORDER = ('f1', 'map', 'precision', 'recall')

Baselines = dict[str, list[tuple[str, float]]]


def _top_k(metric: str) -> Optional[int]:
    match = re.fullmatch(r'[a-z0-9_]+@(\d+)', metric)
    return int(match.group(1)) if match else None


def display_name(metric: str) -> str:
    family = metric.split('@')[0]
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


def positive_share(run: Run) -> Optional[float]:
    if run.task_type in CLASS_LABELED or run.holdout is None:
        return None
    return float(run.holdout['TARGET'].astype(bool).mean())


def baselines(run: Run) -> Baselines:
    out: Baselines = {}
    if run.task_type in CLASS_LABELED:
        return out
    if run.model_type == CLASSIFICATION:
        out['auroc'] = [('chance', 0.5)]
        share = positive_share(run)
        if share is not None:
            out['ap'] = out['auprc'] = [('base rate', share)]
    elif run.model_type == LINK_PREDICTION and run.catalog is not None:
        size, mean_targets, _ = run.catalog
        for metric in run.metric_names:
            k = _top_k(metric)
            if k is not None and metric.startswith('recall@'):
                out[metric] = [('random', k / size)]
            elif k is not None and metric.startswith('precision@'):
                out[metric] = [('random', mean_targets / size)]
    elif run.model_type != LINK_PREDICTION and run.holdout is not None:
        actual = run.holdout['TARGET'].to_numpy(dtype=float)
        for name, pred in (('predict zero', 0.0), ('predict the mean',
                                                   actual.mean())):
            error = pred - actual
            mse = float((error**2).mean())
            for metric, value in (('mae', float(np.abs(error).mean())),
                                  ('mse', mse), ('rmse', math.sqrt(mse))):
                out.setdefault(metric, []).append((name, value))
    return out


def tile_metrics(run: Run) -> list[str]:
    metrics = run.metric_names
    tune = run.tune or (metrics[0] if metrics else None)
    if tune is None:
        return []
    if run.model_type == LINK_PREDICTION:
        k = _top_k(tune)
        order = [tune
                 ] + ([] if k is None else [f'recall@{k}', f'precision@{k}'])
    else:
        order = [tune] + [
            m for m in metrics if not (m == 'auprc' and 'ap' in metrics)
            and not (m == 'mse' and 'rmse' in metrics)
        ]
    return list(dict.fromkeys(order))[:3]


def _tile(run: Run, metric: str, holdout: float, training: Optional[float],
          items: list[tuple[str, float]]) -> dict[str, Any]:
    callout = None
    if run.model_type == LINK_PREDICTION:
        if items and items[0][1] > 0:
            callout = f'{holdout / items[0][1]:.1f}× random'
    elif items:
        from kumo_core.evaluation.suite import MetricStore

        best = max if MetricStore.tune_mode(metric) == 'max' else min
        name, value = best(items, key=lambda item: item[1])
        callout = f'{fmt.signed(holdout - value)} vs {name}'
    return dict(
        name=display_name(metric), tune=metric == run.tune,
        holdout_text=fmt.number(holdout),
        holdout_label=('holdout · before full refit'
                       if run.full_refit else 'holdout'),
        training_text=None if training is None else fmt.number(training),
        training_scope=('Last training epoch; predictions use the selected '
                        'checkpoint.'), gap_text=None if training is None else
        f'holdout minus training: {fmt.signed(holdout - training)}', baselines=[
            dict(text=f'{name} {fmt.number(value)}') for name, value in items
        ], callout=callout)


def _detail(split: dict[str, Any], *dates: str) -> Optional[str]:
    rows = split.get('num_train_rows')
    if not isinstance(rows, int) or not all(split.get(d) for d in dates):
        return None
    return f'{rows:,} rows · ' + ' to '.join(str(split[d])[:10] for d in dates)


def _table(run: Run, holdout: dict[str, float]) -> dict[str, Any]:
    splits = run.splits
    columns = [
        ('Holdout', _detail(splits['test'], 'anchor_date'), holdout),
        ('Validation', _detail(splits['val'],
                               'anchor_date'), _prefixed(run, 'val_')),
        ('Training', _detail(splits['train'], 'min_date',
                             'max_date'), _prefixed(run, 'train_')),
    ]
    metrics = list(run.metric_names)
    if run.model_type == LINK_PREDICTION:
        metrics.sort(key=lambda m: (_RANKING_ORDER.index(m.split('@')[0])
                                    if m.split('@')[0] in _RANKING_ORDER else
                                    len(_RANKING_ORDER), _top_k(m) or 0))
    rows = [(display_name(m), m == run.tune, [v.get(m) for _, _, v in columns])
            for m in metrics]
    if run.model_type == LINK_PREDICTION:
        prefix = 'test_refit_' if _refit(run) else 'test_'
        rows.append(('Locality score', False,
                     [run.metrics.get(prefix + 'locality_score'), None, None]))
    rows = [row for row in rows if any(v is not None for v in row[2])]
    # Drop columns no row fills, such as training values a bundle lacks.
    keep = [i for i in range(3) if any(row[2][i] is not None for row in rows)]
    groups = [
        ('Model before full refit' if run.full_refit else 'Published model',
         int(0 in keep)),
        ('Training run · last epoch', len([i for i in keep if i])),
    ]
    return dict(
        groups=[dict(title=title, span=span) for title, span in groups if span],
        columns=[dict(label=columns[i][0], detail=columns[i][1]) for i in keep],
        rows=[
            dict(
                name=name, tune=tune, texts=[
                    None if values[i] is None else fmt.number(values[i])
                    for i in keep
                ]) for name, tune, values in rows
        ])


def _history(run: Run) -> Optional[dict[str, Any]]:
    """Search-trial scalars with the same tags and epoch labels as the UI."""
    experiments = []
    for trial in mapping(mapping(run.history).get('trial_progress')).values():
        series: dict[str, list[tuple[int, Optional[float]]]] = {}
        for epoch, record in mapping(mapping(trial).get('metrics')).items():
            if not re.fullmatch(r'[0-9]+', epoch):
                continue
            for field, prefix in (('train_metrics', 'train_'),
                                  ('validation_metrics', 'val_')):
                for tag, value in mapping(mapping(record).get(field)).items():
                    if tag.startswith(prefix) and '_per_label_' not in tag:
                        finite = (isinstance(value, (int, float))
                                  and not isinstance(value, bool)
                                  and math.isfinite(value))
                        series.setdefault(tag, []).append(
                            (int(epoch), value if finite else None))
        for tag, points in series.items():
            points.sort(key=lambda point: point[0])
            # The TensorBoard hook adjusts each series independently.
            if points[0][0] == 0:
                series[tag] = [(epoch + 1, value) for epoch, value in points]
        if series:
            experiments.append(series)
    if not experiments:
        return None
    tags = sorted(tag for series in experiments for tag in series)
    metrics = list(dict.fromkeys(tag.split('_', 1)[1] for tag in tags))
    if run.tune in metrics:
        metrics.remove(run.tune)
        metrics.insert(0, run.tune)
    return dict(id='history', label='Epoch history', metrics=metrics,
                experiments=experiments)


def card(run: Run) -> Optional[dict[str, Any]]:
    holdout = _holdout(run)
    training = _prefixed(run, 'train_')
    base = baselines(run)
    tiles = [
        _tile(run, metric, holdout[metric], training.get(metric),
              base.get(metric, [])) for metric in tile_metrics(run)
        if metric in holdout
    ]
    views = [
        dict(
            id='baselines', tiles=tiles,
            label=('Against random' if run.model_type == LINK_PREDICTION else
                   'Against baselines')),
        dict(id='table', label='All metrics', table=_table(run, holdout)),
    ] if tiles else []
    history = _history(run)
    if history is not None:
        views.append(history)
    if not views:
        return None
    return dict(title='Metrics', views=views,
                coverage=run.metrics_coverage() if tiles else None)


def header(run: Run) -> dict[str, Any]:
    """The query and one line about the holdout; either may be missing."""
    frame = run.holdout
    parts = []
    if run.model_type == LINK_PREDICTION:
        rows = run.splits['test'].get('num_train_rows')
        if isinstance(rows, int):
            parts.append(f'{rows:,} {run.entities}')
    elif frame is not None and run.model_type == FORECASTING:
        steps = frame['TIMESTAMP'].nunique()
        word = period(frame['TIMESTAMP'])
        parts.append(f'{frame["ENTITY"].nunique():,} {run.entities} × {steps} '
                     f'{word if steps == 1 else word + "s"} = '
                     f'{len(frame):,} rows')
    elif frame is not None:
        parts.append(f'{len(frame):,} {run.entities}')
    anchor = run.splits['test'].get('anchor_date')
    if parts and anchor:
        parts.append(f'anchor {str(anchor)[:10]}')
    share = positive_share(run)
    if run.model_type == CLASSIFICATION and share is not None:
        parts.append(f'{fmt.percent(share)} True')
    if run.model_type == REGRESSION and frame is not None:
        zero = float((frame['TARGET'] == 0).mean())
        if zero >= 0.1:
            parts.append(f'{fmt.percent(zero)} with actual 0')
    if run.model_type == LINK_PREDICTION and run.catalog is not None:
        size, mean_targets, table = run.catalog
        parts += [
            f'{fmt.number(mean_targets)} items each',
            f'catalog {size:,} {table}'
        ]
    line = ('Holdout: ' + ' · '.join(parts)) if parts else None
    if run.full_refit:
        note = ('Diagnostics and explanations describe the model before full '
                'refit.')
        line = f'{line} · {note}' if line else note
    return dict(query=run.query, holdout=line)
