# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Metrics card, shared by every model type.

Holdout scores describe the evaluated fit, before any full-data refit.
Training and validation scores use the original run's last epoch.
"""

import math
import posixpath
import re
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from kumo.engine._explainability import _files
from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._run import (
    CLASS_LABELED,
    CLASSIFICATION,
    FORECASTING,
    LINK_PREDICTION,
    REGRESSION,
    Run,
)

HIGHER_IS_BETTER = {'auroc', 'ap', 'auprc', 'map', 'recall', 'precision', 'f1'}
LOWER_IS_BETTER = {'mae', 'mse', 'rmse', 'loss'}

_NAMES = {
    'auroc': 'AUROC',
    'ap': 'AP',
    'auprc': 'AUPRC',
    'mae': 'MAE',
    'mse': 'MSE',
    'rmse': 'RMSE',
    'loss': 'Loss',
    'map': 'MAP',
    'recall': 'Recall',
    'precision': 'Precision',
    'f1': 'F1',
    'acc': 'Accuracy',
    'accuracy': 'Accuracy',
    'smape': 'SMAPE',
    'r2': 'R²',
    'mape': 'MAPE',
    'ndcg': 'NDCG',
}
_RANKING_ORDER = ('f1', 'map', 'precision', 'recall')
_LOCALITY = 'locality_score'


def family(metric: str) -> str:
    return metric.split('@')[0]


def top_k(metric: str) -> Optional[int]:
    match = re.fullmatch(r'[a-z0-9_]+@(\d+)', metric)
    return int(match.group(1)) if match else None


def display_name(metric: str) -> str:
    name = _NAMES.get(family(metric), family(metric).upper())
    k = top_k(metric)
    return name if k is None else f'{name}@{k}'


def higher_is_better(metric: str) -> Optional[bool]:
    if family(metric) in HIGHER_IS_BETTER:
        return True
    if family(metric) in LOWER_IS_BETTER:
        return False
    return None


@dataclass(frozen=True)
class Fits:
    """Retained holdout scores and the fit they describe."""

    holdout: dict[str, float]
    refit: bool
    full_refit: bool


def _prefixed(values: Any, prefix: str) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    return {
        key.removeprefix(prefix): float(value)
        for key, value in values.items()
        if key.startswith(prefix) and isinstance(value, (int, float))
        and not isinstance(value, bool) and math.isfinite(value)
    }


def fits(run: Run) -> Fits:
    refit_holdout = _prefixed(run.metrics, 'test_refit_')
    refit = run.files.refit or bool(refit_holdout)
    holdout = refit_holdout if refit else {
        k: v
        for k, v in _prefixed(run.metrics, 'test_').items()
        if not k.startswith('refit_')
    }
    provenance = _files.read_json(
        posixpath.join(run.files.bundle, '_provenance.json'))
    config = _files.read_json(
        posixpath.join(run.files.bundle, 'model', 'config.json'))
    refit_cfg = config.get('refit_cfg') if isinstance(config, dict) else None
    full_refit = (provenance.get('fit_mode') == 'refit_full'
                  if isinstance(provenance, dict) and provenance.get('fit_mode')
                  else isinstance(refit_cfg, dict)
                  and refit_cfg.get('refitted_full') is True)
    return Fits(holdout, refit, full_refit)


# ---------------------------------------------------------------- baselines


@dataclass(frozen=True)
class Baseline:
    name: str
    value: float


def _regression_baselines(run: Run) -> dict[str, list[Baseline]]:
    if run.holdout is None:
        return {}
    actual = run.holdout['TARGET'].to_numpy(dtype=float)
    if actual.size == 0:
        return {}
    out: dict[str, list[Baseline]] = {}
    for name, pred in (('predict zero', np.zeros_like(actual)),
                       ('predict the mean', np.full_like(actual,
                                                         actual.mean()))):
        error = pred - actual
        values = {
            'mae': float(np.abs(error).mean()),
            'mse': float((error**2).mean()),
            'rmse': float(np.sqrt((error**2).mean())),
        }
        for metric, value in values.items():
            out.setdefault(metric, []).append(Baseline(name, value))
    return out


def positive_share(run: Run) -> Optional[float]:
    if run.task_type in CLASS_LABELED:
        return None
    if run.holdout is None or len(run.holdout) == 0:
        return None
    return float(run.holdout['TARGET'].astype(bool).mean())


def _classification_baselines(run: Run) -> dict[str, list[Baseline]]:
    out = {'auroc': [Baseline('chance', 0.5)]}
    share = positive_share(run)
    if share is not None:
        out['ap'] = [Baseline('base rate', share)]
        out['auprc'] = [Baseline('base rate', share)]
    return out


@dataclass(frozen=True)
class Catalog:
    size: int
    mean_targets: float
    table: str


def catalog(run: Run) -> Optional[Catalog]:
    """Catalog size and mean holdout list length for random baselines."""
    files = run.files
    if files.table is None or files.snapshot is None:
        return None
    manifest = _files.read_json(
        posixpath.join(files.table, 'diskgraph_manifest.json'))
    stats = _files.read_json(posixpath.join(files.snapshot,
                                            'ingest_stats.json'))
    if not isinstance(manifest, dict) or not isinstance(stats, dict):
        return None
    rhs = (manifest.get('task_dataset') or {}).get('rhs_node_type')
    counts = (stats.get('stats') or {}).get('node_counts') or {}
    size = counts.get(rhs) if isinstance(rhs, str) else None
    seeds = _files.read_parquet_dir(posixpath.join(files.table, 'test_seeds'),
                                    ['TARGET1'])
    if not isinstance(size,
                      int) or size <= 0 or seeds is None or not len(seeds):
        return None
    lengths = seeds['TARGET1'].map(len).to_numpy(dtype=float)
    return Catalog(size, float(lengths.mean()), str(rhs))


def _random_baselines(run: Run) -> dict[str, list[Baseline]]:
    cat = catalog(run)
    if cat is None:
        return {}
    out: dict[str, list[Baseline]] = {}
    for metric in run.training_job.get('metrics') or []:
        k = top_k(metric)
        if k is None:
            continue
        if family(metric) == 'recall':
            out[metric] = [Baseline('random', k / cat.size)]
        elif family(metric) == 'precision':
            out[metric] = [Baseline('random', cat.mean_targets / cat.size)]
    return out


def baselines(run: Run) -> dict[str, list[Baseline]]:
    if run.model_type == CLASSIFICATION:
        return {} if run.task_type in CLASS_LABELED else _classification_baselines(
            run)
    if run.model_type in (REGRESSION, FORECASTING):
        return _regression_baselines(run)
    return _random_baselines(run)


# --------------------------------------------------------------------- card


def tile_metrics(run: Run) -> list[str]:
    job = run.training_job
    metrics = [m for m in job.get('metrics') or [] if isinstance(m, str)]
    tune = job.get('tune_metric')
    if not isinstance(tune, str):
        tune = metrics[0] if metrics else None
    if tune is None:
        return []
    if run.model_type == LINK_PREDICTION:
        k = top_k(tune)
        order = [tune
                 ] + ([] if k is None else [f'recall@{k}', f'precision@{k}'])
    else:
        names = set(metrics)
        order = [tune] + [
            m for m in metrics if not (m == 'auprc' and 'ap' in names)
            and not (m == 'mse' and 'rmse' in names)
        ]
    seen: list[str] = []
    for metric in order:
        if metric not in seen:
            seen.append(metric)
    return seen[:3]


def _best(metric: str, items: list[Baseline]) -> Optional[Baseline]:
    direction = higher_is_better(metric)
    if not items or direction is None:
        return None
    return (max if direction else min)(items, key=lambda b: b.value)


def tile(run: Run, fit: Fits, metric: str,
         base: dict[str, list[Baseline]]) -> Optional[dict[str, Any]]:
    holdout = fit.holdout.get(metric)
    if holdout is None:
        return None
    training = _prefixed(run.metrics, 'train_').get(metric)
    items = base.get(metric, [])
    callout = None
    if run.model_type == LINK_PREDICTION:
        rand = items[0] if items else None
        if rand is not None and rand.value > 0:
            callout = f'{fmt.multiple(holdout / rand.value)} random'
    else:
        best = _best(metric, items)
        if best is not None:
            callout = f'{fmt.signed(holdout - best.value)} vs {best.name}'
    gap = None if training is None else holdout - training
    return {
        'metric':
        metric,
        'name':
        display_name(metric),
        'tune':
        metric == run.training_job.get('tune_metric'),
        'holdout':
        holdout,
        'holdout_text':
        fmt.number(holdout),
        'holdout_label':
        'holdout · before full refit' if fit.full_refit else 'holdout',
        'training':
        training,
        'training_text':
        None if training is None else fmt.number(training),
        'training_scope':
        'Last training epoch; predictions use the selected checkpoint.',
        'gap':
        gap,
        'gap_text':
        None if gap is None else f'holdout minus training: {fmt.signed(gap)}',
        'baselines': [{
            'name': b.name,
            'value': b.value,
            'text': f'{b.name} {fmt.number(b.value)}'
        } for b in items],
        'callout':
        callout,
    }


def table_metrics(run: Run, fit: Fits) -> list[str]:
    metrics = [
        m for m in run.training_job.get('metrics') or [] if isinstance(m, str)
    ]
    if run.model_type != LINK_PREDICTION:
        return metrics

    def order(metric: str) -> tuple[int, int]:
        fam = family(metric)
        rank = (_RANKING_ORDER.index(fam)
                if fam in _RANKING_ORDER else len(_RANKING_ORDER))
        return rank, top_k(metric) or 0

    return sorted(metrics, key=order)


def _split(stats: Optional[dict[str, Any]], name: str) -> dict[str, Any]:
    value = (stats or {}).get(f'{name}_table_stats')
    return value if isinstance(value, dict) else {}


def _rows(split: dict[str, Any]) -> Optional[int]:
    value = split.get('num_train_rows')
    return value if isinstance(value, int) else None


def _detail_anchor(split: dict[str, Any]) -> Optional[str]:
    rows = _rows(split)
    if rows is None or not split.get('anchor_date'):
        return None
    return f'{fmt.count(rows)} rows · {fmt.date(split["anchor_date"])}'


def _detail_range(split: dict[str, Any]) -> Optional[str]:
    rows = _rows(split)
    if rows is None or not split.get('min_date') or not split.get('max_date'):
        return None
    return (f'{fmt.count(rows)} rows · {fmt.date(split["min_date"])} to '
            f'{fmt.date(split["max_date"])}')


def table(run: Run, fit: Fits) -> dict[str, Any]:
    stats = _files.split_stats(run.files)
    train, val, test = (_split(stats, s) for s in ('train', 'val', 'test'))
    holdout_detail = _detail_anchor(test)
    groups = [{
        'title':
        'Model before full refit' if fit.full_refit else 'Published model',
        'span': 1
    }, {
        'title': 'Training run · last epoch',
        'span': 2
    }]
    columns = [
        {
            'label': 'Holdout',
            'detail': holdout_detail,
            'values': fit.holdout
        },
        {
            'label': 'Validation',
            'detail': _detail_anchor(val),
            'values': _prefixed(run.metrics, 'val_')
        },
        {
            'label': 'Training',
            'detail': _detail_range(train),
            'values': _prefixed(run.metrics, 'train_')
        },
    ]
    tune = run.training_job.get('tune_metric')
    rows = []
    for metric in table_metrics(run, fit):
        values = [column['values'].get(metric) for column in columns]
        if all(v is None for v in values):
            continue
        rows.append({
            'metric':
            metric,
            'name':
            display_name(metric),
            'tune':
            metric == tune,
            'values':
            values,
            'texts': [None if v is None else fmt.number(v) for v in values]
        })
    if run.model_type == LINK_PREDICTION:
        values: list[Optional[float]] = [None] * len(columns)
        holdouts = [i for i, c in enumerate(columns) if c['label'] == 'Holdout']
        sources = [
            f'test_refit_{_LOCALITY}' if fit.refit else f'test_{_LOCALITY}'
        ]
        for index, key in zip(holdouts, sources):
            values[index] = run.metrics.get(key)
        if any(v is not None for v in values):
            rows.append({
                'metric':
                _LOCALITY,
                'name':
                'Locality score',
                'tune':
                False,
                'values':
                values,
                'texts': [None if v is None else fmt.number(v) for v in values]
            })
    # Drop columns no row fills, such as training values a bundle lacks.
    keep = [
        i for i in range(len(columns))
        if any(row['values'][i] is not None for row in rows)
    ]
    start = 0
    kept_groups = []
    for group in groups:
        span = len([i for i in keep if start <= i < start + group['span']])
        if span:
            kept_groups.append({**group, 'span': span})
        start += group['span']
    for row in rows:
        row['values'] = [row['values'][i] for i in keep]
        row['texts'] = [row['texts'][i] for i in keep]
    return {
        'groups':
        kept_groups,
        'columns': [{
            'label': columns[i]['label'],
            'detail': columns[i]['detail']
        } for i in keep],
        'rows':
        rows,
    }


def _history(run: Run) -> Optional[dict[str, Any]]:
    """Search-trial scalars with the same tags and epoch labels as the UI."""
    history = _files.read_json(_files.bundle_file(run.files, 'metric_history'))
    if history is None and run.files.first is not None:
        history = _files.read_json(
            posixpath.join(posixpath.dirname(run.files.first),
                           'auto_train_progress.json'))
    trials = history.get('trial_progress') if isinstance(history,
                                                         dict) else None
    if not isinstance(trials, dict):
        return None
    experiments = []
    tags: set[str] = set()

    # Object.keys in the web client enumerates array indices first, then
    # named trials in their original order.
    def trial_order(key: str) -> int:
        if re.fullmatch(r'0|[1-9][0-9]{0,9}', key):
            return min(int(key), 2**32 - 1)
        return 2**32 - 1

    for key in sorted(trials, key=trial_order):
        trial = trials[key]
        records = trial.get('metrics') if isinstance(trial, dict) else None
        if not isinstance(records, dict):
            continue
        series: dict[str, list[tuple[int, Optional[float]]]] = {}
        for epoch, record in records.items():
            if not re.fullmatch(r'[0-9]{1,10}', epoch) or not isinstance(
                    record, dict):
                continue
            for field, prefix in (('train_metrics', 'train_'),
                                  ('validation_metrics', 'val_')):
                values = record.get(field)
                if not isinstance(values, dict):
                    continue
                for tag, value in values.items():
                    if not tag.startswith(prefix) or '_per_label_' in tag:
                        continue
                    number = value if (
                        isinstance(value, (float, int))
                        and not isinstance(value, bool)
                        and abs(value) <= sys.float_info.max) else None
                    series.setdefault(tag, []).append((int(epoch), number))
        if not series:
            continue
        for tag, points in series.items():
            points.sort(key=lambda point: point[0])
            # The TensorBoard hook adjusts each series independently.
            if points[0][0] == 0:
                series[tag] = [(epoch + 1, value) for epoch, value in points]
        tags.update(series)
        experiments.append(series)
    if not experiments:
        return None
    metrics = list(dict.fromkeys(tag.split('_', 1)[1] for tag in sorted(tags)))
    tune = run.training_job.get('tune_metric')
    if tune in metrics:
        metrics.remove(tune)
        metrics.insert(0, tune)
    return {
        'id': 'history',
        'label': 'Epoch history',
        'metrics': metrics,
        'experiments': experiments
    }


def card(run: Run) -> Optional[dict[str, Any]]:
    fit = fits(run)
    base = baselines(run)
    tiles = [
        t for m in tile_metrics(run)
        if (t := tile(run, fit, m, base)) is not None
    ]
    views = []
    if tiles:
        views.extend([
            {
                'id':
                'baselines',
                'label': ('Against random' if run.model_type == LINK_PREDICTION
                          else 'Against baselines'),
                'tiles':
                tiles
            },
            {
                'id': 'table',
                'label': 'All metrics',
                'table': table(run, fit)
            },
        ])
    history = _history(run)
    if history is not None:
        views.append(history)
    if not views:
        return None
    coverage = run.coverage.line(run.coverage.metrics)
    if run.coverage.holdout_truncated:
        coverage = ' · '.join(
            filter(None, [coverage, run.coverage.holdout_line()]))
    return {
        'title': 'Metrics',
        'views': views,
        'coverage': coverage if tiles else None,
    }
