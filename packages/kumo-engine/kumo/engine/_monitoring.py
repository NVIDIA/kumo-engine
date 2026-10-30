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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from kumo_core.score_counts import Counts
from kumo_core.score_drift import drift_column
from kumo_core.score_drift import score_drift as exact_drift
from kumo_core.summaries import Dataset, Note, Scalar, Summary, Table, Tag

from kumo.engine._card import number
from kumo.engine._manifest import PredictionManifest, TrackingRef

_EVALUATION_FILE = '_evaluation.json'


def _for_model_monitor(files: Any, directory: Path) -> list:
    r"""``files`` with the types Snowflake Model Monitor accepts: a boolean
    ``TARGET`` as 0/1 and timestamps without a time zone (UTC), which load as
    NUMBER and TIMESTAMP_NTZ."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    def monitored(field: Any) -> Any:
        if field.name == 'TARGET' and pa.types.is_boolean(field.type):
            return field.with_type(pa.int8())
        if pa.types.is_timestamp(field.type) and field.type.tz is not None:
            return field.with_type(pa.timestamp(field.type.unit))
        return field

    typed = []
    for path in files:
        source = parquet.ParquetFile(path)
        schema = pa.schema([monitored(field) for field in source.schema_arrow])
        target = str(directory / Path(path).name)
        with parquet.ParquetWriter(target, schema) as writer:
            for batch in source.iter_batches():
                writer.write_table(
                    pa.Table.from_batches([batch]).cast(schema))
        typed.append(target)
    return typed


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics of a prediction run against labels from a newer snapshot.

    ``actuals`` is a Parquet directory of the evaluated rows: ``ENTITY``,
    ``ANCHOR_TIMESTAMP`` (the prediction time for static queries), the
    prediction (``True_PROB`` and ``TARGET_PRED``; ``TARGET_PRED``; or the
    top ``CLASS`` and its ``SCORE``), the label ``TARGET`` and
    ``EVALUATED_AT``, when `evaluate` ran. ``tracking`` is the tracked run
    the result was written into.
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
    tracking: Optional[TrackingRef] = None

    _exported: Any = field(default=None, compare=False, repr=False)

    @property
    def frame(self) -> pd.DataFrame:
        r"""The evaluated rows (``actuals``) as a DataFrame, read into
        memory."""
        if self.actuals is None:
            raise ValueError(f'No evaluated rows: {dict(self.reasons)}')
        return pd.read_parquet(self.actuals)

    def export(self, output_config: Any) -> None:
        r"""Copy the evaluated rows to ``output_config``'s predictions
        destination (a `PredictionOutputConfig`), e.g. a table to monitor.
        A Parquet directory also gets ``_evaluation.json``, the other fields,
        for `load_from_connector`. A Snowflake table gets the types Model
        Monitor needs: ``TARGET`` as a number and TIMESTAMP_NTZ times."""
        import tempfile

        from kumo.engine._scratch import get_scratch_dir
        from kumo.engine.connector import (
            SnowflakeConnector,
            is_file_destination,
        )
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
            for name in ('actuals', 'tracking', '_exported'):
                fields.pop(name)
            extra['sidecars'] = {
                _EVALUATION_FILE: json.dumps(fields, sort_keys=True).encode()
            }
        get_scratch_dir().mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=get_scratch_dir()) as typed:
            if isinstance(output_config.connector, SnowflakeConnector
                          ) and not is_file_destination(
                              output_config.connector, destination):
                files = _for_model_monitor(files, Path(typed))
            output_config._write(destination, files,
                                 export_id=f'{self.prediction_id}-evaluation-'
                                 f'{Path(self.actuals).name}-{written}',
                                 **extra)

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
    exact functions of both sides' score counts over the same bins
    (`kumo_core.score_drift`; ``wasserstein`` is in score units), and None
    when the bins differ. ``features`` compares, per table and column the
    model encodes, the column statistics DiskGraph wrote when each snapshot
    was built (training snapshot, then scoring snapshot). ``tracking`` is
    the tracked run the report was written into.
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
    tracking: Optional[TrackingRef] = None


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


def holdout_evaluation(result: Any,
                       threshold: Optional[float]) -> EvaluationResult:
    r"""``result``'s holdout as evaluated rows with the columns and types
    `evaluate` writes, so its export is a warehouse monitor's baseline."""
    from kumo_core.artifacts import bundle_task_type

    from kumo.engine._scratch import get_scratch_dir

    task_type = bundle_task_type(result.artifact_paths).value
    task = _TASKS.get(task_type)
    if task not in ('binary', 'regression'):
        raise ValueError(f'Holdout evaluated rows cover binary '
                         f'classification and regression, not {task_type}.')
    holdout = result.holdout_df()
    now = pd.Timestamp.now(tz='UTC')
    rows = pd.DataFrame({
        'ENTITY': holdout['ENTITY'],
        'ANCHOR_TIMESTAMP': holdout.get('TIMESTAMP', now.tz_localize(None)),
        'TARGET': holdout['TARGET'],
        'EVALUATED_AT': now,
    }).astype({'ANCHOR_TIMESTAMP': 'datetime64[ns]',
               'EVALUATED_AT': 'datetime64[ns, UTC]'})
    for name in _EMITTED[task]:
        rows[name] = holdout[name]
    if task == 'binary':
        rows['TARGET_PRED'] = (
            rows['True_PROB'].gt(threshold) if threshold is not None else
            pd.Series(pd.NA, index=rows.index, dtype='boolean'))
    actuals = get_scratch_dir() / 'evaluations' / f'holdout-{result.id}'
    shutil.rmtree(actuals, ignore_errors=True)
    actuals.mkdir(parents=True)
    rows.to_parquet(actuals / 'part-00000.parquet', index=False)
    return EvaluationResult(
        prediction_id='holdout', bundle_id=result.id,
        snapshot_id=result.snapshot_id or '', task_type=task_type,
        anchors=(), metrics={}, rows=len(rows), labelled=len(rows),
        labelled_fraction=1.0 if len(rows) else None, matured=True,
        bundle_matches=True, actuals=str(actuals))


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
    from kumo.engine.trainer._provenance import holdout_counts

    reasons: Dict[str, str] = {}
    counts = _score_counts(run)
    model = getattr(getattr(trainer, 'result', None), 'manifest', None)
    holdout = getattr(model, 'holdout', None)
    against = _drift(counts, holdout_counts(model), "the model's holdout",
                     'holdout', reasons)
    earlier = {} if previous is None else _drift(
        counts, _score_counts(previous), 'the previous run', 'previous',
        reasons)

    sides = _snapshot_stats(trainer, run, snapshot, training_snapshot, root)
    totals = [side['counts'] for side in sides.values()]
    input_changes = None
    if all(isinstance(total, dict) for total in totals):
        input_changes = _count_changes(*totals)
    else:
        reasons['input_changes'] = _missing('ingest statistics', sides,
                                            'counts')
    run_bundle = getattr(run, 'bundle_id', None)
    return DriftReport(
        prediction_id=getattr(run, 'id', None),
        bundle_id=run_bundle,
        score=None if counts is None else drift_column(counts),
        psi=against.get('psi'),
        ks=against.get('ks'),
        wasserstein=against.get('wasserstein'),
        holdout_truncated=None if holdout is None else holdout.truncated,
        bundle_matches=_same(run_bundle, _bundle_id(trainer)),
        previous_prediction_id=getattr(previous, 'id', None),
        previous_psi=earlier.get('psi'),
        previous_ks=earlier.get('ks'),
        previous_wasserstein=earlier.get('wasserstein'),
        input_changes=input_changes,
        reasons=reasons,
        features=_features(trainer, sides, reasons),
    )


def _score_counts(run: Any) -> Optional[Counts]:
    stats = (getattr(getattr(run, 'manifest', None), 'stats', None)
             or {}).get('predictions')
    counts = getattr(stats, 'counts', None)
    return None if counts is None else Counts.from_dict(counts)


def _drift(counts: Optional[Counts], reference: Optional[Counts], name: str,
           side: str, reasons: Dict[str, str]) -> Dict[str, float]:
    r"""The exact drift of ``counts`` from ``reference``, ``name``'s counts;
    empty, with why in ``reasons[side]``, unless both counted the same score
    on the same bins."""
    if counts is None:
        reasons[side] = 'the run records no score counts'
    elif reference is None:
        reasons[side] = f'{name} records no score counts'
    else:
        drift = exact_drift(counts, reference)
        if drift is not None:
            return drift
        reasons[side] = (f'the run and {name} did not count the same score '
                         f'on the same bins')
    return {}


def track(report: Any, run: Any, settings: Any, trainer: Any,
          started: datetime, labels: Any = None, previous: Any = None) -> Any:
    r"""``report``, a monitoring call of ``run`` that started at ``started``
    (an evaluation against the ``labels`` snapshot, or drift since the
    ``previous`` run), with the ``tracking``
    of the run the tracker's ``log_monitoring`` wrote it into; a failure
    warns, or raises under ``on_error='raise'``."""
    tracking = None if settings is None else settings.tracking
    if tracking is None:
        return report

    from kumo.engine._mlops import full_error, raise_on_error, warn_failure
    from kumo.engine._tracking import log_run

    try:
        record = _record(report, run, tracking, trainer, started, labels,
                         previous)
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
    run_id = getattr(tracking, 'run_id', None)
    if failure is not None or not run_id:
        return report
    return replace(report, tracking=TrackingRef(
        platform=str(getattr(tracking, 'platform', type(tracking).__name__)),
        run_id=str(run_id), url=getattr(tracking, 'run_url', None)))


#: The prediction run's tags a monitoring record carries, which name it.
_IDENTITY = ('kumo.prediction_id', 'kumo.bundle_id', 'kumo.as_of',
             'kumo.model_name', 'kumo.model_version')


def _record(report: Any, run: Any, tracking: Any, trainer: Any,
            started: datetime, labels: Any, previous: Any) -> Any:
    r"""``report`` as summaries, tagged like ``run``'s prediction run, which
    the tracker writes them into or names a run after."""
    from kumo.engine._tracking import RunSpec, prediction_record, run_record

    evaluating = isinstance(report, EvaluationResult)
    manifest = getattr(run, 'manifest', None)
    identity: Mapping[str, Optional[str]] = {
        'kumo.prediction_id': getattr(run, 'id', None),
        'kumo.bundle_id': getattr(run, 'bundle_id', None)}
    label = None
    if manifest is not None:
        prediction = prediction_record(manifest, model=(
            None if report.bundle_matches is False else
            getattr(trainer, 'result', None)))
        identity, label = prediction.tags, prediction.model_label
    platform = getattr(tracking, 'platform', None)
    ref = getattr(run, 'tracking', None)
    tracked = ref is not None and ref.platform == platform
    earlier = getattr(previous, 'tracking', None)
    record = run_record(RunSpec(
        kind='monitoring',
        parents={'monitor': 'evaluate' if evaluating else 'drift'},
        summaries=(_evaluation_summaries(report, labels) if evaluating else
                   _drift_summaries(report, (
                       earlier.run_name if earlier is not None
                       and earlier.platform == platform else None)))))
    return replace(
        record, tags={**record.tags, **{
            name: value for name, value in identity.items()
            if name in _IDENTITY and value}},
        model_label=label, started=started,
        finished=datetime.now(timezone.utc),
        prediction_run=ref.run_id if tracked else None)


def _drift_summaries(report: 'DriftReport',
                     previous_name: Optional[str] = None) -> List[Summary]:
    r"""What a tracker logs of ``report``: drift since training and since
    the previous run, named ``previous_name`` (else by its id), and, when the
    run's model is the trainer's, the input changes and column drift since
    training (its score drift since training is the prediction run's own);
    notes on what is missing and why."""
    summaries: List[Summary] = []
    reasons = report.reasons
    previous = {'psi': report.previous_psi, 'ks': report.previous_ks,
                'wasserstein': report.previous_wasserstein}
    if report.previous_prediction_id:
        summaries.append(Tag('kumo.drift_previous',
                             report.previous_prediction_id))
    if report.previous_psi is not None:
        summaries += [Scalar(f'drift/{name}_prev', value)
                      for name, value in previous.items()]
    sides = [] if report.bundle_matches is False else [
        ('training', report.psi, report.ks)]
    sides.append((previous_name or report.previous_prediction_id,
                  report.previous_psi, report.previous_ks))
    drift = [f"vs {side}: {_values({'psi': psi, 'ks': ks})}"
             for side, psi, ks in sides if psi is not None]
    if drift:
        summaries.append(Note('notes', f"Drift {' · '.join(drift)}."))
    if report.bundle_matches is False:
        against = {'psi': report.psi, 'ks': report.ks,
                   'wasserstein': report.wasserstein}
        summaries.append(Note('notes', (
            "This trainer's model did not score this run, so its drift "
            'since training is not logged here'
            + (f'; against its holdout: {_values(against)}'
               if report.psi is not None else '') + '.')))
        reasons = {name: reason for name, reason in reasons.items()
                   if name == 'previous'}
    else:
        summaries += _input_changes(report.input_changes or {})
        summaries += _feature_drift(report.features)
        if report.holdout_truncated:
            summaries.append(Note('notes', "The model's holdout file was "
                                  'truncated to fit its size limit.'))
    return summaries + _reason_notes(reasons)


def _evaluation_summaries(result: EvaluationResult,
                          labels: Any) -> List[Summary]:
    r"""What a tracker logs of ``result``: its metrics as ``realized/<m>``,
    the labelled share, the ``labels`` snapshot (as a dataset when stored)
    and notes on maturity and on what is missing."""
    from kumo.engine._manifest import artifact_summaries

    manifest = getattr(labels, 'manifest', None)
    datasets = [] if manifest is None else [
        replace(summary, context='labels')
        for summary in artifact_summaries(manifest, 'labels')
        if isinstance(summary, Dataset) and summary.name == manifest.id]
    summaries: List[Summary] = [
        Tag('kumo.labels_snapshot_id', result.snapshot_id), *datasets,
        *(Scalar(f'realized/{name}', value)
          for name, value in result.metrics.items() if value is not None),
    ]
    if result.labelled_fraction is not None:
        summaries.append(Scalar('realized/labelled_share',
                                result.labelled_fraction))
    summaries.append(Note('notes', f'Labelled {result.labelled:,} of '
                          f'{result.rows:,} scored rows with '
                          f'`{result.snapshot_id}`.'))
    if result.matured is False:
        summaries.append(Note('notes', 'Labels are not mature yet: the '
                              'label window ends after the snapshot data.'))
    return summaries + _reason_notes(result.reasons)


_REASONS = {'holdout': 'Drift since training', 'previous':
            'Drift since the previous run', 'input_changes': 'Input changes',
            'features': 'Column drift', 'metrics': 'Realized metrics',
            'matured': 'Label maturity'}
_TOP_FEATURES = 3


def _reason_notes(reasons: Mapping[str, str]) -> List[Summary]:
    return [Note('notes', f'{_REASONS.get(name, name)}: {reason}.')
            for name, reason in reasons.items()]


def _values(values: Mapping[str, Optional[float]]) -> str:
    return ' · '.join(f'{name} {number(value)}'
                      for name, value in values.items() if value is not None)


def _input_changes(changes: Mapping[str, Tuple[Any, Any]]) -> List[Summary]:
    if not changes:
        return []
    rows = tuple(
        (key.removeprefix('tables.').removesuffix('.rows'), before, after,
         (after - before) / before * 100 if before and after is not None
         else None)
        for key, (before, after) in changes.items())
    brief = len(rows) <= 12
    return [Table('drift/input_changes',
                  ('table or link', 'at training', 'now', 'change %'), rows,
                  brief=brief),
            *([] if brief else [Note('notes', f'{len(rows)} inputs changed '
                                     'size: `drift/input_changes.json`.')])]


def _feature_drift(features: Mapping[str, Mapping[str, Mapping[str, Any]]]
                   ) -> List[Summary]:
    rows = sorted(
        ((table, column, change['type'] if 'type_changed' not in change else
          ' → '.join(map(str, change['type_changed'])), change.get('psi'),
          change.get('standardized_mean_shift'),
          (change.get('delta') or {}).get('null_rate'))
         for table, columns in features.items()
         for column, change in columns.items()),
        key=lambda row: (row[3] is None, -(row[3] or 0), -abs(row[4] or 0)))
    if not rows:
        return []
    moved = [f'`{table}.{column}` (psi {number(psi)})' if psi is not None
             else f'`{table}.{column}` (mean shift {number(shift)} sd)'
             for table, column, _, psi, shift, _ in rows
             if psi is not None or shift is not None][:_TOP_FEATURES]
    return [Table('drift/features', ('table', 'column', 'type', 'psi',
                                     'mean shift', 'null-rate change'),
                  tuple(rows)),
            Note('notes', f"Most moved columns: {', '.join(moved)}; all in "
                 '`drift/features.json`.' if moved else
                 'Column drift: `drift/features.json`.')]


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


_NUMERIC_STATS = ('mean', 'std', 'min', 'max', 'q25', 'q50', 'q75',
                  'skewness', 'kurtosis', 'null_rate', 'num_unique')
_CATEGORY_STATS = ('null_rate', 'num_unique')
_TIME_STATS = ('min_secs', 'max_secs')
_TEXT_STATS = ('null_rate', 'num_unique', 'avg_tokens', 'max_tokens',
               'glove_overlap')
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
        absent, uncompared = [], []
        for table, columns in sorted(used.items()):
            for column in sorted(columns):
                old = (before.get(table) or {}).get(column)
                new = (after.get(table) or {}).get(column)
                if old is None or new is None:
                    absent.append(f'{table}.{column}')
                    continue
                change = _column_change(old, new)
                if change is None:
                    uncompared.append(
                        f"{table}.{column} ({new.get('semantic_type')})")
                else:
                    features.setdefault(table, {})[column] = change
        notes = []
        if absent:
            notes.append('column statistics missing on a side for ' +
                         ', '.join(absent))
        if uncompared:
            notes.append('no drift statistics for the type of ' +
                         ', '.join(uncompared))
        if notes:
            reasons['features'] = '; '.join(notes)
        return features
    except Exception as error:
        reasons['features'] = f'feature statistics failed: {error}'
        return {}


def _parent_snapshot_id(manifest: Any) -> Optional[str]:
    parents = getattr(manifest, 'parents', None)
    if getattr(parents, 'derived_from', None) is not None:
        return _parent_snapshot_id(parents.derived_from)
    table = getattr(parents, 'training_table', None)
    snapshot = getattr(getattr(table, 'parents', None), 'snapshot', None)
    if snapshot is None:
        snapshot = getattr(parents, 'snapshot', None)
    return getattr(snapshot, 'id', None)


_ARGUMENTS = {'training': 'training_snapshot', 'scoring': 'snapshot'}
_NAMED_BY = {'training': "the model's manifest", 'scoring': "the run's manifest"}


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
        snapshot_id = _parent_snapshot_id(manifest)
        read = _reader(passed, snapshot_id, root)
        ingest = read('ingest_stats.json')
        stats = ingest.get('stats') if isinstance(ingest, dict) else None
        sides[side] = {
            'named': passed is not None or snapshot_id is not None,
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
    unnamed = [_NAMED_BY[side] for side in missing if not sides[side]['named']]
    if unnamed:
        return (f"no {what} for the {' and '.join(missing)} snapshot: "
                f"{' and '.join(unnamed)} name{'s' * (len(unnamed) == 1)} "
                f"no snapshot; pass {passes}")
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
    elif kind == 'text':
        change = _deltas(old, new, _TEXT_STATS)
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
