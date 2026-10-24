# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Realized performance and score drift of prediction runs."""

import dataclasses
import json
import math
import shutil
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from kumo.engine._manifest import PredictionManifest


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics of a prediction run against labels from a newer snapshot.

    ``actuals`` is a Parquet directory of the evaluated rows: ``ENTITY``,
    ``ANCHOR_TIMESTAMP`` (the prediction time for static queries), the
    prediction (``True_PROB`` and ``TARGET_PRED``; ``TARGET_PRED``; or the
    top ``CLASS`` and its ``SCORE``) and the label ``TARGET``.
    """

    prediction_id: str
    bundle_id: Optional[str]
    snapshot_id: str
    task_type: Optional[str]
    anchors: Tuple[str, ...]
    metrics: Mapping[str, Optional[float]]
    rows: int
    labelled: int
    labelled_fraction: Optional[float]
    matured: Optional[bool]
    bundle_matches: Optional[bool]
    reasons: Mapping[str, str] = field(default_factory=dict)
    actuals: Optional[str] = None

    def export(self, output_config: Any) -> None:
        r"""Copy the evaluated rows to ``output_config``'s predictions
        destination (a `PredictionOutputConfig`), e.g. a table to monitor."""
        if self.actuals is None:
            raise ValueError(f'No evaluated rows: {dict(self.reasons)}')
        sink = output_config.connector.sink_backend
        destination = output_config.destinations()['predictions']
        files = sorted(str(path) for path in Path(self.actuals).glob('*.parquet'))
        sink.write_files(output_config.connector, destination, files,
                         output_config.write_mode,
                         export_id=f'{self.prediction_id}-evaluation-'
                         f'{Path(self.actuals).name}')


@dataclass(frozen=True)
class DriftReport:
    """Score and input drift of a prediction run; ``reasons`` explains Nones.

    ``psi``, ``ks`` and ``wasserstein`` compare the run's scores with the
    holdout's; ``previous_*`` with ``previous_prediction_id``'s run. All are
    computed from the recorded percentile curves (``wasserstein`` is in score
    units).
    """

    prediction_id: Optional[str]
    bundle_id: Optional[str]
    score: Optional[str]
    psi: Optional[float]
    ks: Optional[float]
    wasserstein: Optional[float]
    holdout_truncated: Optional[bool]
    bundle_matches: Optional[bool]
    previous_prediction_id: Optional[str]
    previous_psi: Optional[float]
    previous_ks: Optional[float]
    previous_wasserstein: Optional[float]
    input_changes: Optional[Mapping[str, Tuple[Any, Any]]]
    reasons: Mapping[str, str]


def evaluate(trainer: Any, run: Any, snapshot: Any) -> EvaluationResult:
    query = trainer._training_query
    if query is None:
        raise RuntimeError('Fit or load the Trainer before evaluate()')
    reasons: Dict[str, str] = {}
    out: Dict[str, Any] = dict(task_type=None, anchors=(), metrics={},
                               rows=0, labelled=0, matured=None,
                               reasons=reasons, actuals=None)
    try:
        _evaluate(out, trainer, run, snapshot, query)
    except Exception as error:
        reasons['metrics'] = f"{out['step']} failed: {error}"
    out.pop('step')
    return EvaluationResult(
        prediction_id=run.id,
        bundle_id=run.bundle_id,
        snapshot_id=snapshot.id,
        labelled_fraction=(out['labelled'] /
                           out['rows'] if out['rows'] else None),
        bundle_matches=_same(run.bundle_id, trainer.bundle_id),
        **out,
    )


_TASKS = {
    'binary_classification': 'binary',
    'regression': 'regression',
    'multiclass_classification': 'multiclass',
    'link_prediction': 'link',
    'temporal_link_prediction': 'link',
    'static_link_prediction': 'link',
}


def _evaluate(out: Dict[str, Any], trainer: Any, run: Any, snapshot: Any,
              query: str) -> None:
    import diskgraph
    from kumo_core.pquery import _target_maximum
    from kumo_core.training import custom_table
    from kumo_core.types import PredictionArtifactType
    from pyarrow import parquet

    from kumo.engine.pquery.predictive_query import PredictiveQuery

    reasons = out['reasons']
    out['step'] = 'parsing the query'
    pquery = PredictiveQuery.from_snapshot(snapshot, query)
    parsed = pquery._parsed
    static = parsed.query_type == 'static'
    out['task_type'] = parsed.task_type

    out['step'] = 'reading the predictions'
    files = [
        str(path) for path in replace(
            run, artifact_type=PredictionArtifactType.PREDICTIONS).files
    ]
    columns = parquet.read_schema(files[0]).names
    manifest = run.manifest
    anchors = () if static or manifest is None else tuple(
        sorted({_ns(value) for value in manifest.prediction_table.anchors}))
    keys = ['ENTITY']
    if len(anchors) > 1:
        keys.append('ANCHOR_TIMESTAMP')
    out['anchors'] = tuple(
        pd.Timestamp(anchor, tz='UTC').isoformat() for anchor in anchors)
    if anchors:
        window_end = parsed.query.get_combined_date_offset_range(
        ).end_date_offset
        try:
            latest = _target_maximum(parsed, str(snapshot.index_path), min)
            out['matured'] = bool(
                pd.Timestamp(max(anchors)) + window_end <= pd.Timestamp(latest))
        except Exception as error:
            reasons['matured'] = str(error)
    task = _TASKS.get(parsed.task_type)
    if task is None:
        reasons['metrics'] = f'{parsed.task_type} metrics are not supported'
        return
    if not static and not anchors:
        reasons['metrics'] = 'the run records no anchor timestamps'
        return
    if 'ANCHOR_TIMESTAMP' in keys and 'ANCHOR_TIMESTAMP' not in columns:
        reasons['metrics'] = ('the run has several anchors but its '
                              'predictions have no ANCHOR_TIMESTAMP column')
        return

    out['step'] = 'generating labels'
    entity, timestamp, target = (custom_table.entity_column(),
                                 custom_table.timestamp_column(),
                                 custom_table.target_column())
    actuals = _actuals_dir(run, snapshot)
    try:
        with pquery._label_parts(
                snapshot, anchors,
                trainer.training_table_generation_plan) as parts:
            out['step'] = 'joining labels with predictions'
            label_columns = parquet.read_schema(parts[0]).names
            shutil.rmtree(actuals, ignore_errors=True)
            emitted = [name for name in _EMITTED[task] if name in columns]
            spec = {
                'task': task,
                'prediction_keys': keys,
                'label_keys': [entity, timestamp][:len(keys)],
                'label_target': target,
                'actuals_dir': str(actuals),
                'actual_columns': [[entity, 'ENTITY']] +
                ([[timestamp, 'ANCHOR_TIMESTAMP']]
                 if timestamp in label_columns and not static else []) +
                [[target, 'TARGET']],
                'prediction_columns': [[name, name] for name in emitted],
                'constant_timestamp': (['ANCHOR_TIMESTAMP',
                                        _prediction_time(manifest)]
                                       if static and manifest is not None
                                       else None),
                'spill_dir': str(actuals.parent / f'.spill-{uuid4().hex}'),
            }
            joined = json.loads(
                diskgraph.evaluate_prediction_parquet(files, parts,
                                                      json.dumps(spec)))
    except Exception as error:
        if (out['step'] != 'generating labels'
                or 'empty training table' not in str(error)):
            raise
        reasons['metrics'] = 'no entity has a label at these anchors'
        return
    out.update(rows=joined['rows'], labelled=joined['labelled'],
               actuals=str(actuals))
    if out['labelled'] == 0:
        reasons['metrics'] = 'no prediction has a label'
        return
    metrics = out['metrics'] = {
        name: _finite(value)
        for name, value in joined['metrics'].items()
    }
    if None in metrics.values():
        reasons['metrics'] = 'a metric is undefined for these labels'


_EMITTED = {
    'binary': ('True_PROB', 'TARGET_PRED'),
    'regression': ('TARGET_PRED', ),
    'multiclass': ('CLASS', 'SCORE'),
    'link': (),
}


def _prediction_time(manifest: PredictionManifest) -> int:
    anchors = manifest.prediction_table.anchors
    return _ns(anchors[0] if anchors else manifest.timings.started_at)


def _actuals_dir(run: Any, snapshot: Any) -> Path:
    from kumo.engine._scratch import get_scratch_dir

    selection = getattr(run, 'selection', None)
    root = (Path(selection.pointer_path).parent if selection is not None
            else get_scratch_dir() / 'evaluations' / str(run.id))
    return root / 'actuals' / str(snapshot.id).replace('/', '_')


def score_drift(trainer: Any, run: Any, *, previous: Any, snapshot: Any,
                training_snapshot: Any) -> DriftReport:
    reasons: Dict[str, str] = {}
    score, summary = _score_summary(run)
    model = getattr(trainer, 'manifest', None)
    holdout = getattr(model, 'holdout', None)
    baseline = None
    if summary is None:
        reasons['holdout'] = 'the run records no output score statistics'
    elif holdout is None:
        reasons['holdout'] = 'the model records no holdout score statistics'
    else:
        baseline = holdout.columns.get(score)
        if _curves(summary, baseline) is None:
            reasons['holdout'] = f'the holdout records no percentiles for {score}'

    earlier = None
    if previous is not None:
        previous_score, earlier = _score_summary(previous)
        if summary is None or earlier is None:
            reasons['previous'] = 'a run records no score statistics'
        elif previous_score != score:
            reasons['previous'] = 'the runs record different scores'
            earlier = None
        elif _curves(summary, earlier) is None:
            reasons['previous'] = 'a run records no percentiles'

    input_changes = None
    try:
        counts = [s._counts for s in (training_snapshot, snapshot)]
    except Exception:
        reasons['input_changes'] = 'both snapshots are needed'
    else:
        input_changes = _count_changes(*counts)
    run_bundle = getattr(run, 'bundle_id', None)
    stats = (_psi, _ks, _wasserstein)
    against = [[None if b is None or summary is None else stat(summary, b)
                for stat in stats] for b in (baseline, earlier)]
    return DriftReport(
        prediction_id=getattr(run, 'id', None),
        bundle_id=run_bundle,
        score=score,
        psi=against[0][0],
        ks=against[0][1],
        wasserstein=against[0][2],
        holdout_truncated=None if holdout is None else holdout.truncated,
        bundle_matches=_same(run_bundle, getattr(trainer, 'bundle_id', None)),
        previous_prediction_id=getattr(previous, 'id', None),
        previous_psi=against[1][0],
        previous_ks=against[1][1],
        previous_wasserstein=against[1][2],
        input_changes=input_changes,
        reasons=reasons,
    )


def track(report: Any, run: Any, settings: Any) -> None:
    r"""Log ``report`` as a new monitoring run of ``run``; best-effort."""
    tracking = None if settings is None else settings.tracking
    if tracking is None:
        return
    from kumo.engine._scratch import new_run_id
    from kumo.engine._tracking import RunRecord, log_run

    try:
        values = {**report.metrics} if isinstance(
            report, EvaluationResult) else {}
        for name, value in dataclasses.asdict(report).items():
            if name == 'input_changes':
                for key, (before, after) in (value or {}).items():
                    values[f'input.{key}.before'] = before
                    values[f'input.{key}.after'] = after
            elif name not in ('metrics', 'reasons', 'actuals'):
                values[name] = value
        prediction_id = getattr(run, 'id', None)
        record = RunRecord(
            name=f'monitor-{prediction_id}',
            params={
                key: str(value)
                for key, value in values.items()
                if isinstance(value, str)
            },
            metrics={
                key: float(value)
                for key, value in values.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(value)
            },
            tags={
                key: str(value)
                for key, value in (
                    ('kumo.kind', 'monitoring'),
                    ('kumo.run_id', f'{prediction_id}-monitor-{new_run_id()}'),
                    ('kumo.prediction_run', prediction_id),
                    ('kumo.bundle_id', getattr(run, 'bundle_id', None)),
                ) if value is not None
            },
        )
    except Exception as error:
        warnings.warn(f'Could not track the monitoring run: {error}',
                      UserWarning, stacklevel=3)
        return
    log_run(tracking, 'log_monitoring', record)


def _count_changes(before: Mapping[str, Mapping[str, int]],
                   after: Mapping[str, Mapping[str, int]]
                   ) -> Dict[str, Tuple[Any, Any]]:
    changes = {}
    for kind, key in (('rows', 'tables.{}.rows'), ('edges', 'edges.{}')):
        old, new = before.get(kind) or {}, after.get(kind) or {}
        for name in sorted(set(old) | set(new)):
            if not name.endswith('_reverse') and old.get(name) != new.get(name):
                changes[key.format(name)] = (old.get(name), new.get(name))
    return changes


def _score_summary(run: Any) -> Tuple[Optional[str], Optional[Mapping]]:
    try:
        columns = run.manifest.stats['predictions'].columns
        return next(iter(columns.items()))
    except Exception:
        return None, None


def _psi(actual: Mapping[str, Any],
         expected: Optional[Mapping[str, Any]]) -> Optional[float]:
    r"""PSI over ``expected``'s deciles; each side's bin shares are read off
    its percentile curve (linear between min, q01..q99 and max)."""
    curves = _curves(actual, expected)
    if curves is None:
        return None
    edges = np.unique(curves[1][0][10:91:10])

    def shares(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
        upper = np.searchsorted(values, edges, side='right')
        lower = np.maximum(upper - 1, 0)
        upper = np.minimum(upper, len(values) - 1)
        width = values[upper] - values[lower]
        step = np.divide(edges - values[lower], width,
                         out=np.ones_like(edges), where=width > 0)
        cdf = levels[lower] + np.clip(step, 0, 1) * (levels[upper] -
                                                     levels[lower])
        return np.maximum(np.diff(np.concatenate([[0], cdf, [1]])), 1e-6)

    actual_shares, expected_shares = shares(*curves[0]), shares(*curves[1])
    return _finite(
        np.sum((actual_shares - expected_shares) *
               np.log(actual_shares / expected_shares)))


def _ks(a: Mapping[str, Any], b: Optional[Mapping[str, Any]]) -> Optional[float]:
    r"""sup |F_a - F_b| of the percentile curves, at every knot from both
    sides (point masses at repeated knots)."""
    curves = _curves(a, b)
    if curves is None:
        return None
    points = np.union1d(curves[0][0], curves[1][0])

    def cdf(values: np.ndarray, levels: np.ndarray, side: str) -> np.ndarray:
        index = np.searchsorted(values, points, side=side)
        lower = np.clip(index - 1, 0, len(values) - 1)
        upper = np.clip(index, 0, len(values) - 1)
        width = values[upper] - values[lower]
        step = np.divide(points - values[lower], width,
                         out=np.zeros_like(points), where=width > 0)
        inside = levels[lower] + step * (levels[upper] - levels[lower])
        return np.where(index == 0, 0.0,
                        np.where(index == len(values), 1.0, inside))

    return _finite(
        max(
            np.max(np.abs(cdf(*curves[0], side) - cdf(*curves[1], side)))
            for side in ('left', 'right')))


def _wasserstein(a: Mapping[str, Any],
                 b: Optional[Mapping[str, Any]]) -> Optional[float]:
    r"""Integral of |Q_a - Q_b| over levels, exact for the linear curves."""
    curves = _curves(a, b)
    if curves is None:
        return None
    d = curves[0][0] - curves[1][0]
    left, right = d[:-1], d[1:]
    size = np.abs(left) + np.abs(right)
    crossing = np.divide(left**2 + right**2, 2 * size, out=np.zeros_like(size),
                         where=size > 0)
    segments = np.where(left * right >= 0, size / 2, crossing)
    return _finite(0.01 * np.sum(segments))


def _curves(a: Mapping[str, Any],
            b: Optional[Mapping[str, Any]]) -> Optional[tuple]:
    try:
        curves = (_percentile_curve(a), _percentile_curve(b))
    except Exception:
        return None
    return None if None in curves else curves


def _percentile_curve(summary: Optional[Mapping[str, Any]]) -> Optional[tuple]:
    percentiles = (summary or {}).get('percentiles')
    if not percentiles or len(percentiles) != 99:
        return None
    values = np.maximum.accumulate(
        np.array([summary['min'], *percentiles, summary['max']], dtype=float))
    return values, np.linspace(0, 1, 101)


def _ns(value: Any) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.value


def _same(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    return None if a is None or b is None else a == b


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None
