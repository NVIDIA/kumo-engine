# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Realized performance and score drift of prediction runs."""

import json
import math
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics of a prediction run against labels from a newer snapshot.

    ``actuals`` is a Parquet directory of the labels joined to the run's
    predictions (``ENTITY``, ``ANCHOR_TIMESTAMP`` when generated, ``TARGET``).
    """

    run_id: str
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


@dataclass(frozen=True)
class DriftReport:
    """Score and input drift of a prediction run; ``reasons`` explains Nones."""

    run_id: Optional[str]
    bundle_id: Optional[str]
    score: Optional[str]
    psi: Optional[float]
    holdout_truncated: Optional[bool]
    bundle_matches: Optional[bool]
    previous_run_id: Optional[str]
    previous_psi: Optional[float]
    input_changes: Optional[Mapping[str, Tuple[Any, Any]]]
    reasons: Mapping[str, str]


def evaluate(trainer: Any, run: Any, snapshot: Any) -> EvaluationResult:
    query = trainer.training_query
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
        run_id=run.run_id,
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
    anchors = () if static else _anchors(files, columns, run.manifest)
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
            spec = {
                'task': task,
                'prediction_keys': keys,
                'label_keys': [entity, timestamp][:len(keys)],
                'label_target': target,
                'actuals_dir': str(actuals),
                'actual_columns': [[entity, 'ENTITY']] +
                ([[timestamp, 'ANCHOR_TIMESTAMP']]
                 if timestamp in label_columns else []) + [[target, 'TARGET']],
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


def _actuals_dir(run: Any, snapshot: Any) -> Path:
    from kumo.engine._scratch import get_scratch_dir

    selection = getattr(run, 'selection', None)
    root = (Path(selection.pointer_path).parent if selection is not None
            else get_scratch_dir() / 'evaluations' / str(run.run_id))
    return root / 'actuals' / str(snapshot.id).replace('/', '_')


def score_drift(trainer: Any, run: Any, *, previous: Any, snapshot: Any,
                training_snapshot: Any) -> DriftReport:
    reasons: Dict[str, str] = {}
    score, summary = _score_summary(run)
    if summary is None:
        reasons['psi'] = 'the run records no output score statistics'
    provenance = getattr(trainer._training_result, 'provenance', None)
    holdout = (provenance or {}).get('holdout')
    psi = None
    if not holdout:
        reasons['psi'] = 'the bundle records no holdout score statistics'
    elif summary is not None:
        baseline = (holdout.get('columns') or {}).get(score)
        psi = _psi(summary, baseline)
        if psi is None:
            reasons['psi'] = f'the holdout records no percentiles for {score}'

    previous_psi = None
    if previous is not None:
        previous_score, previous_summary = _score_summary(previous)
        if summary is None or previous_summary is None:
            reasons['previous_psi'] = 'a run records no score statistics'
        elif previous_score != score:
            reasons['previous_psi'] = 'the runs record different scores'
        else:
            previous_psi = _psi(summary, previous_summary)
            if previous_psi is None:
                reasons['previous_psi'] = 'a run records no percentiles'

    input_changes = None
    try:
        counts = [s.counts for s in (training_snapshot, snapshot)]
    except Exception:
        reasons['input_changes'] = 'both snapshots are needed'
    else:
        input_changes = _count_changes(*counts)
    run_bundle = getattr(run, 'bundle_id', None)
    return DriftReport(
        run_id=getattr(run, 'run_id', None),
        bundle_id=run_bundle,
        score=score,
        psi=psi,
        holdout_truncated=(holdout or {}).get('truncated'),
        bundle_matches=_same(run_bundle, getattr(trainer, 'bundle_id', None)),
        previous_run_id=getattr(previous, 'run_id', None),
        previous_psi=previous_psi,
        input_changes=input_changes,
        reasons=reasons,
    )


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
        columns = run.manifest['stats']['predictions']['columns']
        return next(iter(columns.items()))
    except Exception:
        return None, None


def _psi(actual: Mapping[str, Any],
         expected: Optional[Mapping[str, Any]]) -> Optional[float]:
    r"""PSI over ``expected``'s deciles; each side's bin shares are read off
    its percentile curve (linear between min, q01..q99 and max)."""
    try:
        curves = [_percentile_curve(actual), _percentile_curve(expected)]
    except Exception:
        return None
    if None in curves:
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


def _percentile_curve(summary: Optional[Mapping[str, Any]]) -> Optional[tuple]:
    percentiles = (summary or {}).get('percentiles')
    if not percentiles or len(percentiles) != 99:
        return None
    values = np.maximum.accumulate(
        np.array([summary['min'], *percentiles, summary['max']], dtype=float))
    return values, np.linspace(0, 1, 101)


def _anchors(files: list, columns: list,
             manifest: Optional[Mapping[str, Any]]) -> Tuple[int, ...]:
    values = ((manifest or {}).get('prediction_table') or {}).get('anchors')
    if not values and 'ANCHOR_TIMESTAMP' in columns:
        import pyarrow.compute as pc
        from pyarrow import parquet

        values = set()
        for path in files:
            for batch in parquet.ParquetFile(path).iter_batches(
                    columns=['ANCHOR_TIMESTAMP']):
                values.update(
                    pc.unique(batch.column(0).drop_null()).to_pylist())
    return tuple(sorted({_ns(value) for value in values or []}))


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
