# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Realized performance and score drift of prediction runs."""

import dataclasses
import json
import math
import shutil
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from kumo.engine._manifest import PredictionManifest

_EVALUATION_FILE = '_evaluation.json'


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics of a prediction run against labels from a newer snapshot.

    ``actuals`` is a Parquet directory of the evaluated rows: ``ENTITY``,
    ``ANCHOR_TIMESTAMP`` (the prediction time for static queries), the
    prediction (``True_PROB`` and ``TARGET_PRED``; ``TARGET_PRED``; or the
    top ``CLASS`` and its ``SCORE``), the label ``TARGET`` and
    ``EVALUATED_AT``, when `evaluate` ran.
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

    _exported: Any = field(default=None, compare=False, repr=False)

    def export(self, output_config: Any) -> None:
        r"""Copy the evaluated rows to ``output_config``'s predictions
        destination (a `PredictionOutputConfig`), e.g. a table to monitor.
        A Parquet directory also gets ``_evaluation.json``, the other fields,
        for `load_from_connector`."""
        from kumo.engine.trainer.trainer import _holds_parquet

        if self.actuals is None:
            raise ValueError(f'No evaluated rows: {dict(self.reasons)}')
        destination = output_config._targets(
            self.prediction_id)['predictions']
        files = sorted(str(path) for path in Path(self.actuals).glob('*.parquet'))
        written = max(Path(path).stat().st_mtime_ns for path in files)
        extra: Dict[str, Any] = {}
        columns = output_config._id_columns(destination, self.prediction_id,
                                            self.bundle_id)
        if columns:
            extra['extra_columns'] = columns
        if _holds_parquet(output_config.connector, destination):
            fields = dataclasses.asdict(self)
            for name in ('actuals', '_exported'):
                fields.pop(name)
            extra['sidecars'] = {
                _EVALUATION_FILE: json.dumps(fields, sort_keys=True).encode()
            }
        output_config._write(destination, files,
                             export_id=f'{self.prediction_id}-evaluation-'
                             f'{Path(self.actuals).name}-{written}', **extra)

    @classmethod
    def load_from_connector(cls, connector: Any, *,
                            path: str) -> 'EvaluationResult':
        r"""Load an evaluation `export` wrote to a Parquet directory; its rows
        are downloaded into ``actuals``, removed with the result."""
        import tempfile
        import weakref

        from kumo_core.sources.transfer import StagedFile, fetch_all
        from pyarrow.fs import FileSelector, FileType

        from kumo.engine._scratch import get_scratch_dir
        from kumo.engine.connector import file_store

        store = file_store(connector, path)
        if store is None:
            raise TypeError(f'{path} is not a file destination of this '
                            f'connector')
        filesystem, root = store
        root = root.rstrip('/')
        try:
            with filesystem.open_input_stream(
                    f'{root}/{_EVALUATION_FILE}') as reader:
                fields = json.loads(reader.read())
        except (FileNotFoundError, OSError) as error:
            raise FileNotFoundError(
                f'{path} has no {_EVALUATION_FILE}: not an evaluation export'
            ) from error
        if filesystem.get_file_info(
                f'{root}/_SUCCESS').type == FileType.NotFound:
            raise FileNotFoundError(
                f'{path} has no _SUCCESS marker: the export is incomplete or '
                f'still running')
        scratch = get_scratch_dir()
        scratch.mkdir(parents=True, exist_ok=True)
        local = Path(tempfile.mkdtemp(prefix='fetched-evaluation-',
                                      dir=scratch))

        def fetch(staged: StagedFile, target: str) -> None:
            with filesystem.open_input_stream(staged.path) as reader, \
                    open(target, 'wb') as writer:
                shutil.copyfileobj(reader, writer, 1 << 20)

        fetch_all([
            StagedFile(info.path, info.size)
            for info in filesystem.get_file_info(FileSelector(root))
            if info.type == FileType.File and info.path.endswith('.parquet')
        ], str(local / 'actuals'), fetch=fetch)
        fields['anchors'] = tuple(fields['anchors'])
        result = cls(**fields, actuals=str(local / 'actuals'),
                     _exported=local)
        weakref.finalize(result, shutil.rmtree, str(local), True)
        return result


@dataclass(frozen=True)
class DriftReport:
    """Score and input drift of a prediction run; ``reasons`` explains Nones.

    ``psi``, ``ks`` and ``wasserstein`` compare the run's scores with the
    holdout's; ``previous_*`` with ``previous_prediction_id``'s run. All are
    computed from the recorded percentile curves (``wasserstein`` is in score
    units). ``features`` compares, per table and column the model encodes,
    the column statistics DiskGraph wrote when each snapshot was built
    (training snapshot, then scoring snapshot).
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
    features: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict)


def evaluate(trainer: Any, run: Any, snapshot: Any) -> EvaluationResult:
    query = trainer._training_query
    if query is None:
        raise RuntimeError('Fit or load the Trainer before evaluate()')
    reasons: Dict[str, str] = {}
    out: Dict[str, Any] = dict(task_type=None, anchors=(), metrics={},
                               rows=0, labelled=0, matured=None,
                               reasons=reasons, actuals=None,
                               step='importing')
    try:
        _evaluate(out, trainer, run, snapshot, query)
    except Exception as error:
        reasons['metrics'] = f"{out['step']} failed: {error}"
    out.pop('step')
    if out['matured'] is False:
        warnings.warn(
            f'Prediction run {run.id}: labels are not mature yet, since the '
            'label window ends after the snapshot data; the metrics cover '
            'only the outcomes observed so far.', UserWarning, stacklevel=3)
    return EvaluationResult(
        prediction_id=run.id,
        bundle_id=run.bundle_id,
        snapshot_id=snapshot.id,
        labelled_fraction=(out['labelled'] /
                           out['rows'] if out['rows'] else None),
        bundle_matches=_same(run.bundle_id, _bundle_id(trainer)),
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
    pquery = PredictiveQuery._from_snapshot(snapshot, query)
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
    spill = actuals.parent / f'.spill-{uuid4().hex}'
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
                'constant_timestamps': ([[
                    'ANCHOR_TIMESTAMP', _prediction_time(manifest)
                ]] if static and manifest is not None else []) +
                [['EVALUATED_AT', time.time_ns()]],
                'spill_dir': str(spill),
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
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    out.update(rows=joined['rows'], labelled=joined['labelled'],
               actuals=str(actuals) if joined['labelled'] else None)
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


def _bundle_id(trainer: Any) -> Optional[str]:
    result = getattr(trainer, 'result', None)
    return None if result is None else result.id


def _actuals_dir(run: Any, snapshot: Any) -> Path:
    from kumo.engine._scratch import get_scratch_dir

    selection = getattr(run, 'selection', None)
    root = (Path(selection.pointer_path).parent if selection is not None
            else get_scratch_dir() / 'evaluations' / str(run.id))
    return root / 'actuals' / str(snapshot.id).replace('/', '_')


def score_drift(trainer: Any, run: Any, *, previous: Any, snapshot: Any,
                training_snapshot: Any, root: Any = None) -> DriftReport:
    reasons: Dict[str, str] = {}
    score, summary = _score_summary(run)
    result = getattr(trainer, 'result', None)
    model = None if result is None else result.manifest
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

    sides = _snapshot_stats(trainer, run, snapshot, training_snapshot, root)
    counts = [side['counts'] for side in sides.values()]
    input_changes = None
    if all(isinstance(count, dict) for count in counts):
        input_changes = _count_changes(*counts)
    else:
        reasons['input_changes'] = _missing('ingest statistics', sides,
                                            'counts')
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
        bundle_matches=_same(run_bundle, _bundle_id(trainer)),
        previous_prediction_id=getattr(previous, 'id', None),
        previous_psi=against[1][0],
        previous_ks=against[1][1],
        previous_wasserstein=against[1][2],
        input_changes=input_changes,
        reasons=reasons,
        features=_features(trainer, sides, reasons),
    )


def track(report: Any, run: Any, settings: Any) -> None:
    r"""Log ``report`` as a new monitoring run of ``run``; a failure warns,
    or raises under ``on_error='raise'``."""
    tracking = None if settings is None else settings.tracking
    if tracking is None:
        return

    from kumo.engine._mlops import full_error, raise_on_error, warn_failure
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
            elif name == 'features':
                values.update(feature_metrics(value or {}))
            elif name not in ('metrics', 'reasons', 'actuals'):
                values[name] = value
        prediction_id = getattr(run, 'id', None)
        record = RunRecord(
            name=(f'evaluate-{prediction_id}' if isinstance(
                report, EvaluationResult) else f'drift-{prediction_id}'),
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
            artifacts=({
                'feature_drift.json': json.dumps(report.features, indent=2,
                                                 sort_keys=True, default=str)
            } if getattr(report, 'features', None) else {}),
        )
    except Exception as error:
        if settings.on_error == 'warn':
            warn_failure(
                f'Could not track the monitoring run: {full_error(error)}',
                error, 3)
        failure: Optional[BaseException] = error
    else:
        failure = log_run(tracking, 'log_monitoring', record)
    raise_on_error(settings, failure, f'Monitoring of prediction run '
                   f'{getattr(run, "id", None)}', stacklevel=3)


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


#: Columns per ranked list, and tables, in a monitoring run's metrics; keeps
#: the run far below Snowflake's 200 metrics per run.
FEATURE_METRICS_TOP = 10
_NUMERIC_STATS = ('mean', 'std', 'min', 'max', 'q25', 'q50', 'q75',
                  'skewness', 'kurtosis', 'null_rate', 'num_unique')
_CATEGORY_STATS = ('null_rate', 'num_unique')
_TIME_STATS = ('min_secs', 'max_secs')
_SMOOTHING = 1e-6


def _features(trainer: Any, sides: Mapping[str, Mapping[str, Any]],
              reasons: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    r"""Per-column statistic changes between the build-time ``col_stats.json``
    of the training and scoring snapshots, for the columns the model
    encodes. Never raises; gaps go to ``reasons['features']``."""
    try:
        before, after = (side['col_stats'] for side in sides.values())
        if not (isinstance(before, dict) and isinstance(after, dict)):
            reasons['features'] = _missing('column statistics', sides,
                                           'col_stats')
            return {}
        used = _used_columns(getattr(trainer, 'result', None))
        if used is None:
            reasons['features'] = 'the model records no encoded columns'
            return {}
        features: Dict[str, Dict[str, Any]] = {}
        absent = []
        for table, columns in sorted(used.items()):
            for column in sorted(columns):
                old = (before.get(table) or {}).get(column)
                new = (after.get(table) or {}).get(column)
                if old is None or new is None:
                    absent.append(f'{table}.{column}')
                    continue
                change = _column_change(old, new)
                if change is not None:
                    features.setdefault(table, {})[column] = change
        if absent:
            reasons['features'] = ('column statistics missing on a side for '
                                   + ', '.join(absent))
        return features
    except Exception as error:
        reasons['features'] = f'feature statistics failed: {error}'
        return {}


def _parent_snapshot_id(manifest: Any) -> Optional[str]:
    parents = getattr(manifest, 'parents', None)
    table = getattr(parents, 'training_table', None)
    snapshot = getattr(getattr(table, 'parents', None), 'snapshot', None)
    if snapshot is None:
        snapshot = getattr(parents, 'snapshot', None)
    return getattr(snapshot, 'id', None)


_ARGUMENTS = {'training': 'training_snapshot', 'scoring': 'snapshot'}


def _snapshot_stats(trainer: Any, run: Any, snapshot: Any,
                    training_snapshot: Any,
                    root: Any) -> Dict[str, Dict[str, Any]]:
    r"""The build-time ``col_stats.json`` and ingest counts of the training
    and scoring snapshots: those passed, else those the manifests name, from
    the local snapshot store or the artifact ``root``. Never raises."""
    model = getattr(getattr(trainer, 'result', None), 'manifest', None)
    sides = {}
    for side, passed, manifest in (
        ('training', training_snapshot, model),
        ('scoring', snapshot, getattr(run, 'manifest', None)),
    ):
        read = _reader(passed, _parent_snapshot_id(manifest), root)
        ingest = read('ingest_stats.json')
        stats = ingest.get('stats') if isinstance(ingest, dict) else None
        sides[side] = {
            'col_stats': read('col_stats.json'),
            'counts': {
                'rows': dict(stats.get('source_rows_by_table', {})),
                'edges': dict(stats.get('edge_counts', {})),
            } if isinstance(stats, dict) else None,
        }
    return sides


def _reader(snapshot: Any, snapshot_id: Optional[str],
            root: Any) -> Callable[[str], Optional[Any]]:
    index = getattr(snapshot, 'index_path', None)
    if index is None and snapshot_id:
        index = _local_index(snapshot_id)
    if index is not None:
        def local(name: str) -> Optional[Any]:
            try:
                return json.loads((Path(index) / name).read_text())
            except (OSError, ValueError):
                return None
        return local
    if root is None or not snapshot_id:
        return lambda name: None
    from kumo.engine import _snapshot_io

    try:
        path = root.location('snapshots/' + snapshot_id.replace('@', '/'))
        generation, _ = _snapshot_io.read(root.connector, path)
        filesystem, _ = _snapshot_io._location(root.connector, path)
    except Exception:
        return lambda name: None

    def stored(name: str) -> Optional[Any]:
        try:
            return _snapshot_io._read_json(filesystem,
                                           f'{generation}/index/{name}')
        except Exception:
            return None
    return stored


def _missing(what: str, sides: Mapping[str, Mapping[str, Any]],
             name: str) -> str:
    missing = [side for side, stats in sides.items()
               if not isinstance(stats[name], dict)]
    passes = ', '.join(
        f'{_ARGUMENTS[side]}=GraphSnapshot.load_from_connector(store, '
        f'path=...)' for side in missing)
    return (f"no {what} for the {' and '.join(missing)} snapshot: pass "
            f"{passes}, or turn MLOps on with the artifact root that stores "
            f"{'them' if len(missing) > 1 else 'it'}")


def _local_index(snapshot_id: str) -> Optional[Path]:
    from kumo.engine._scratch import get_scratch_dir

    graph_key, _, version = snapshot_id.partition('@')
    if not graph_key or not version:
        return None
    root = get_scratch_dir() / 'snapshots' / graph_key
    found = sorted(path for path in root.glob(f'{version}*')
                   if (path / 'col_stats.json').is_file())
    return found[0] if len(found) == 1 else None


def _used_columns(result: Any) -> Optional[Dict[str, set]]:
    from kumo_core.encoders.config import EncodersConfig
    from kumo_core.io.json import load_json

    path = (getattr(result, 'artifact_paths', None) or {}).get('encoders')
    if not path or not Path(path).is_file():
        return None
    config = load_json(path, EncodersConfig)
    if config is None:
        return None
    return {table: set(columns) for table, columns in config.encoders.items()}


def _column_change(old: Mapping[str, Any],
                   new: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    kind = new.get('semantic_type')
    if old.get('semantic_type') != kind:
        return {'type': kind, 'type_changed': [old.get('semantic_type'), kind]}
    if kind == 'numerical':
        change = _deltas(old, new, _NUMERIC_STATS)
        std = old.get('std')
        shift = change['delta'].get('mean')
        change['standardized_mean_shift'] = (
            shift / std if shift is not None and std else None)
    elif kind in ('categorical', 'multicategorical'):
        change = _deltas(old, new, _CATEGORY_STATS)
        p, q = _distribution(old), _distribution(new)
        change['psi'] = _category_psi(p, q)
        change['js'] = _js(p, q)
        change['category_counts_capped'] = bool(
            old.get('category_counts_capped')
            or new.get('category_counts_capped'))
    elif kind == 'timestamp':
        change = _deltas(old, new, _TIME_STATS)
    else:
        return None
    return {'type': kind, **change}


def _stat(stats: Mapping[str, Any], name: str) -> Optional[float]:
    if name == 'null_rate':
        count = stats.get('count')
        return stats.get('null_count', 0) / count if count else None
    if name == 'num_unique':
        value = stats.get('num_unique', stats.get('num_unique_multi'))
    else:
        value = stats.get(name)
    return float(value) if isinstance(value, (int, float)) and not isinstance(
        value, bool) and math.isfinite(value) else None


def _deltas(old: Mapping[str, Any], new: Mapping[str, Any],
            names: Tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {'training': {}, 'scoring': {}, 'delta': {},
                           'relative': {}}
    for name in names:
        before, after = _stat(old, name), _stat(new, name)
        out['training'][name], out['scoring'][name] = before, after
        if before is None or after is None:
            continue
        out['delta'][name] = after - before
        if before:
            out['relative'][name] = (after - before) / abs(before)
    return out


def _distribution(stats: Mapping[str, Any]) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for value, count in stats.get('category_counts') or ():
        counts[str(value)] = counts.get(str(value), 0.0) + float(count)
    total = sum(counts.values())
    return {key: count / total for key, count in counts.items()} if total \
        else {}


def _smoothed(p: Mapping[str, float], q: Mapping[str, float]
              ) -> Tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys]) + _SMOOTHING
    b = np.array([q.get(k, 0.0) for k in keys]) + _SMOOTHING
    return a / a.sum(), b / b.sum()


def _category_psi(p: Mapping[str, float],
                  q: Mapping[str, float]) -> Optional[float]:
    if not p or not q:
        return None
    a, b = _smoothed(p, q)
    return float(np.sum((b - a) * np.log(b / a)))


def _js(p: Mapping[str, float], q: Mapping[str, float]) -> Optional[float]:
    if not p or not q:
        return None
    a, b = _smoothed(p, q)
    m = (a + b) / 2
    return float((np.sum(a * np.log2(a / m)) + np.sum(b * np.log2(b / m))) / 2)


def _metric_name(*parts: str) -> str:
    import re
    return '.'.join(re.sub(r'[^A-Za-z0-9_-]', '_', part) for part in parts)


def feature_metrics(features: Mapping[str, Mapping[str, Mapping[str, Any]]]
                    ) -> Dict[str, float]:
    r"""A bounded summary of ``features`` for a monitoring run: the
    `FEATURE_METRICS_TOP` columns by absolute standardized mean shift and by
    PSI, and the largest of each per table (for the same number of
    tables)."""
    metrics: Dict[str, float] = {}
    for key, name in (('standardized_mean_shift', 'shift'), ('psi', 'psi')):
        ranked = sorted(
            ((abs(change[key]), table, column)
             for table, columns in features.items()
             for column, change in columns.items()
             if isinstance(change.get(key), float)
             and math.isfinite(change[key])),
            reverse=True)
        for value, table, column in ranked[:FEATURE_METRICS_TOP]:
            metrics[_metric_name('features', name, table, column)] = value
        tables: Dict[str, float] = {}
        for value, table, _ in ranked:
            tables.setdefault(table, value)
        for table, value in sorted(tables.items(), key=lambda item: -item[1]
                                   )[:FEATURE_METRICS_TOP]:
            metrics[_metric_name('features', f'max_{name}', table)] = value
    return metrics
