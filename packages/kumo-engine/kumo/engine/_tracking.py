# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Platform-neutral run records and the `Tracking` protocol."""

import dataclasses
import json
import logging
import os
import queue
import re
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.instrumentation import format_error

from kumo.engine._events import Event, _finite_floats
from kumo.engine._mlops import full_error, warn_failure
from kumo.engine._text import entity, number
from kumo.engine._version import __version__

if TYPE_CHECKING:
    from kumo_core.summaries import Summary
    from kumoapi.plugin import ModelRegistryLocation, PublishedModel

    from kumo.engine._manifest import (
        ModelManifest,
        PredictionManifest,
        TrackingRef,
    )

logger = logging.getLogger(__name__)

DROPPED_PARAMS = 'kumo.dropped_params'
DROPPED_METRICS = 'kumo.dropped_metrics'
QUEUE_SIZE = 1000
FINISH_SECONDS = 30.0
MODEL_SECONDS = 3600.0


@dataclass(frozen=True)
class RunRecord:
    r"""One tracked run; ``steps`` are ``(step, value)`` histories."""

    name: Optional[str] = None
    params: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    steps: Mapping[str, Tuple[Tuple[int, float],
                              ...]] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, Mapping[str, Optional[str]]] = field(
        default_factory=dict)
    summaries: Tuple['Summary', ...] = ()
    description: Optional[str] = None
    model_label: Optional[str] = None
    started: Optional[datetime] = None
    finished: Optional[datetime] = None
    prediction_run: Optional[str] = None
    name_suffix: Optional[str] = None


@dataclass(frozen=True)
class RunSpec:
    r"""What a run records; ``parents`` become ``kumo.<name>`` tags."""

    kind: str
    name: Optional[str] = None
    params: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    steps: Mapping[str, Sequence[Tuple[int, float]]] = field(
        default_factory=dict)
    tags: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, Mapping[str, Optional[str]]] = field(
        default_factory=dict)
    parents: Mapping[str, Optional[str]] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    summaries: Sequence['Summary'] = ()
    status: Optional[str] = None
    error: Optional[str] = None
    description: Optional[str] = None
    model_label: Optional[str] = None
    started: Optional[datetime] = None
    finished: Optional[datetime] = None
    name_suffix: Optional[str] = None


INPUT_FIELDS = ('connector', 'location', 'version', 'version_kind')


def fit_inputs(manifest: Any) -> Dict[str, Dict[str, Optional[str]]]:
    r"""Source tables of a training table or model manifest."""
    try:
        table, kind = manifest, getattr(manifest, 'kind', None)
        if kind == 'model':
            table = getattr(_parents(manifest), 'training_table', None)
        parents = None if kind == 'prediction' else _parents(table)
        snapshot = _parents(
            getattr(_parents(manifest) if kind == 'prediction' else parents,
                    'snapshot', None))
        tables = dict(getattr(snapshot, 'source_tables', None) or {})
        source = getattr(parents, 'source_table', None)
        if source is not None:
            tables['training_table'] = source
        return {
            str(name): {key: _text(getattr(entry, key))
                        for key in INPUT_FIELDS}
            for name, entry in tables.items()
        }
    except Exception as error:
        logger.warning('Could not read the fit\'s source tables: %s',
                       format_error(error), exc_info=_debug())
        return {}


def _parents(manifest: Any) -> Any:
    if manifest is None or getattr(manifest, 'reconstructed', False):
        return None
    return getattr(manifest, 'parents', None)


def _text(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def model_metrics(record: RunRecord) -> Dict[str, float]:
    r"""A run's metrics and its exact ``role='model'`` scalars."""
    from kumo_core.summaries import Scalar

    return {**record.metrics,
            **{summary.name: summary.value for summary in record.summaries
               if isinstance(summary, Scalar) and summary.role == 'model'}}


def metric_name(metric: str) -> str:
    r"""The plan's acc is logged as accuracy."""
    return 'accuracy' if metric == 'acc' else metric


def logged_name(name: str) -> str:
    r"""``val_auroc`` as ``val/auroc``, ``best_val_auroc`` as
    ``val/best_auroc``."""
    best = name.startswith('best_')
    split, _, metric = name.removeprefix('best_').partition('_')
    if split not in ('train', 'val') or not metric:
        return name
    return f"{split}/{'best_' if best else ''}{metric_name(metric)}"


def logged_names(metrics: Mapping[str, Any],
                 refit_full: bool = False) -> Dict[str, Any]:
    r"""Tracker names of metrics; test/<m> is the shipped model's holdout."""
    refit = any(name.startswith('test_refit_') for name in metrics)
    named = {}
    for name, value in metrics.items():
        if not name.startswith('test_'):
            named[logged_name(name)] = value
            continue
        metric = name.removeprefix('test_')
        latest = metric.startswith('refit_') == refit
        metric = metric_name(metric.removeprefix('refit_'))
        if not latest and refit_full:
            continue
        named[f'test/{metric}' if latest and not refit_full else
              f'before_refit/{metric}'] = value
    return named


def model_label(registered: Optional[str], query: Optional[str],
                task_type: Optional[str]) -> str:
    r"""The registered name's last part, else <entity>-<task type>."""
    if registered:
        return registered.rsplit('.', 1)[-1]
    return '-'.join(filter(None, (entity(query), task_type))) or 'model'


def by_step(steps: Mapping[str, Sequence[Tuple[int, float]]]
            ) -> List[Tuple[int, Dict[str, float]]]:
    r"""Each step's metrics, in step order."""
    grouped: Dict[int, Dict[str, float]] = {}
    for name, history in steps.items():
        for step, value in history:
            grouped.setdefault(step, {})[name] = value
    return sorted(grouped.items())


_TRIAL_PARENTS = ('snapshot_id', 'task_type')


def run_record(spec: RunSpec) -> RunRecord:
    params = {key: _param_value(value) for key, value in spec.params.items()}
    return RunRecord(
        name=spec.name,
        params=params,
        metrics=logged_names(_finite_floats(spec.metrics)),
        steps={logged_name(name): tuple(history)
               for name, history in spec.steps.items()},
        tags={
            **_user_tags(spec.tags, params),
            'kumo.kind': spec.kind,
            'kumo.engine_version': __version__,
            **{f'kumo.{name}': str(value)
               for name, value in spec.parents.items() if value},
            **({} if spec.status is None else {'kumo.status': spec.status}),
            **({} if spec.error is None else {'kumo.error': spec.error}),
        },
        artifacts=dict(spec.artifacts),
        inputs=dict(spec.inputs),
        summaries=tuple(spec.summaries),
        description=spec.description,
        model_label=spec.model_label,
        started=spec.started,
        finished=spec.finished,
        name_suffix=spec.name_suffix,
    )


def _user_tags(tags: Optional[Mapping[str, Any]],
               params: Mapping[str, str]) -> Dict[str, str]:
    dropped = sorted(key for key in tags or {}
                     if key.startswith('kumo.') or key in params)
    if dropped:
        warnings.warn(
            f'Tracking tags {dropped} are reserved and not recorded.',
            UserWarning, stacklevel=3)
    return {
        key: str(value)
        for key, value in (tags or {}).items() if key not in dropped
    }


def trial_spec(events: Sequence[Event]) -> RunSpec:
    r"""One trial attempt of a fit, from its events so far."""
    started = next(event for event in events if event.name == 'trial_started')
    trial, attempt = started.trial, int(started.params['attempt'])
    metrics: Dict[str, float] = {}
    steps: Dict[str, List[Tuple[int, float]]] = {}
    ended: Dict[str, Any] = {}
    for event in events:
        if event.name == 'epoch_ended' and event.epoch is not None:
            for name, value in event.metrics.items():
                steps.setdefault(name, []).append((event.epoch, value))
        elif event.name == 'trial_ended':
            ended = dict(status=str(event.params.get('status')),
                         error=None if 'error' not in event.params else str(
                             event.params['error']))
            metrics.update({
                f'best_{name}': value
                for name, value in event.metrics.items() if name != 'seconds'
            })
    return RunSpec(
        kind='trial',
        name=f'trial-{trial}-attempt-{attempt}',
        params=dict(started.params),
        metrics=metrics,
        steps=steps,
        **ended,
    )


def prediction_record(manifest: 'PredictionManifest',
                      error: Optional[BaseException] = None,
                      model: Any = None,
                      summaries: Sequence['Summary'] = ()) -> RunRecord:
    r"""A prediction run's record, from its manifest."""
    from kumo.engine._prep_cost import prep_summaries
    from kumo.engine.trainer._provenance import holdout_counts

    table = manifest.prediction_table
    params = {**manifest.scoring.to_dict(), 'anchors': table.anchors}
    model_manifest = getattr(model, 'manifest', None)
    trained = getattr(_parents(model_manifest or manifest.parents.model),
                      'training_table', None)
    registered = getattr(model, 'registered_model', None)
    label = model_label(getattr(registered, 'name', None),
                        getattr(trained, 'query', None),
                        getattr(trained, 'task_type', None))
    as_of = _as_of(manifest)
    suffix = f'-{as_of}' + ('' if table.anchors else
                            f'-{_started(manifest):%H%M%S}')
    timings = manifest.timings
    window = getattr(getattr(model_manifest, 'holdout', None), 'window', None)
    return run_record(
        RunSpec(
            kind='prediction',
            name=run_name('prediction', (
                f'{label}-{manifest.parents.model.id[:8]}' if registered is None
                else versioned(label, registered.version)), suffix),
            name_suffix=suffix,
            model_label=label,
            started=_started(manifest),
            finished=(None if timings is None or not timings.finished_at
                      else _utc(timings.finished_at)),
            params={
                key: value
                for key, value in params.items() if value not in (None, 'none')
            },
            parents={
                'bundle_id': manifest.parents.model.id,
                'prediction_id': manifest.id,
                'as_of': as_of,
                'entity': entity(getattr(trained, 'query', None)),
                'tune_metric': _tune_metric(model),
                'labels_mature': _labels_mature(manifest, window),
                **prediction_parents(manifest),
                'fit_run_id': fit_run_id(model_manifest),
                **registered_parents(model),
            },
            artifacts={
                PROVENANCE_FILE:
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True,
                           default=str)
            },
            inputs=fit_inputs(manifest),
            summaries=(*_prediction_summaries(
                manifest, holdout_counts(model_manifest), window), *summaries,
                *prep_summaries(manifest.parents.snapshot)),
            error=None if error is None else format_error(error),
        ))


def prediction_parents(manifest: Optional['PredictionManifest']
                       ) -> Mapping[str, Any]:
    r"""Pointer tags to the scoring snapshot and task type."""
    if manifest is None:
        return {}
    training_table = getattr(_parents(manifest.parents.model),
                             'training_table', None)
    return {'snapshot_id': manifest.parents.snapshot.id,
            'task_type': getattr(training_table, 'task_type', None)}


def run_name(kind: str, core: str, suffix: Optional[str] = None) -> str:
    r"""<kind>-<core><suffix>, the kind not repeated."""
    word = _KIND_WORDS.get(kind, kind)
    named = re.match(rf'{word}([-_]|$)', core, re.IGNORECASE)
    return (core if named else f'{word}-{core}') + (suffix or '')


_KIND_WORDS = {'prediction': 'predict'}


def versioned(model: str, version: str) -> str:
    r"""<model>-v<version>, or <model>-<version> for a named version."""
    return f'{model}-v{version}' if version.isdigit() else f'{model}-{version}'


def fit_run_id(manifest: Any) -> Optional[str]:
    r"""The tracker's run of the fit a `ModelManifest` records."""
    tracking = getattr(manifest, 'tracking', None)
    return None if tracking is None else tracking.run_id


def registered_parents(model: Any) -> Mapping[str, Any]:
    r"""The model's registered name and version, not its alias."""
    from kumo.engine.trainer._run_spec import _registered

    return {} if model is None else _registered(model.registered_model)


def _started(manifest: 'PredictionManifest') -> datetime:
    timings = manifest.timings
    return (datetime.now(timezone.utc) if timings is None else
            _utc(timings.started_at))


def _as_of(manifest: 'PredictionManifest') -> str:
    r"""The anchor date (UTC), or the scored snapshot's creation date."""
    dates = sorted({_utc(anchor).date().isoformat()
                    for anchor in manifest.prediction_table.anchors})
    if not dates:
        created = manifest.parents.snapshot.created_at
        dates = [(_utc(created) if created else _started(manifest)
                  ).date().isoformat()]
    return dates[0] if len(dates) == 1 else f'{dates[0]}..{dates[-1]}'


def _tune_metric(model: Any) -> Optional[str]:
    r"""The metric ``model``, a `TrainingResult`, was tuned on, if known."""
    plan = getattr(getattr(model, '_trainer', None), 'model_plan', None)
    metric = getattr(getattr(plan, 'training_job', None), 'tune_metric', None)
    text = None if metric is None else str(getattr(metric, 'value', metric))
    return None if text in (None, '', '???') else metric_name(text)


def _one_epoch_note(spec: RunSpec,
                    held: Optional[Mapping[str, float]]) -> RunSpec:
    r"""Single-point series moved into a note."""
    from kumo_core.summaries import Note

    single = {name: points[0][1] for name, points in spec.steps.items()
              if len(points) < 2}
    values = {**{logged_name(name): value for name, value in single.items()},
              **(held or {})}
    values.pop('train/epoch_seconds', None)
    if not single and held is None:
        return spec
    text = ' · '.join(f"{name.replace('/', ' ')} {number(value)}"
                      for name, value in values.items())
    return dataclasses.replace(
        spec, steps={name: points for name, points in spec.steps.items()
                     if name not in single},
        summaries=(*spec.summaries, Note('notes',
                                         f'Training (1 epoch): {text}')))


def _labels_mature(manifest: 'PredictionManifest',
                   window: Optional[Sequence[Any]]) -> Optional[str]:
    r"""When the latest anchor's labels are all known (UTC)."""
    import pandas as pd

    anchors = manifest.prediction_table.anchors
    if not window or not anchors:
        return None
    _, end, step, unit = window
    stats = (manifest.stats or {}).get('predictions')
    arrays = getattr(stats, 'counts', None) or {}
    last = max((int(name.removeprefix('TARGET_PRED_h'))
                for name in arrays.get('arrays', {})
                if name.startswith('TARGET_PRED_h')), default=1)
    latest = max(_utc(anchor) for anchor in anchors)
    return (latest + pd.DateOffset(**{unit: end + (last - 1) * step})
            ).date().isoformat()


def _utc(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return (moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else
            moment.astimezone(timezone.utc))


def _prediction_summaries(manifest: 'PredictionManifest', baseline: Any,
                          window: Optional[Sequence[Any]] = None
                          ) -> List['Summary']:
    r"""Row counts, skipped rows and score summaries with drift."""
    from kumo_core.prediction_counts import prediction_summaries
    from kumo_core.score_counts import Counts
    from kumo_core.summaries import Note, Scalar

    table = manifest.prediction_table
    summaries: List['Summary'] = []
    if table.rows is not None:
        summaries.append(Scalar('pred/rows', table.rows))
    if table.skipped_rows is not None:
        requested = table.requested_rows
        summaries.append(Scalar('pred/skipped_share', table.skipped_rows /
                                requested if requested else 0.0))
        if table.skipped_rows:
            reason = f': {table.skip_reason}' if table.skip_reason else ''
            summaries.append(Note(
                'notes', f'{table.skipped_rows:,} requested rows were not '
                f'scored{reason}.'))
    stats = (manifest.stats or {}).get('predictions')
    if stats is not None and stats.counts is not None:
        summaries += prediction_summaries(
            Counts.from_dict(stats.counts),
            threshold=manifest.scoring.binary_classification_threshold,
            baseline=baseline, timeframe=None if window is None else tuple(
                window))
    return summaries


@contextmanager
def system_metrics(tracking: Any) -> Iterator[None]:
    r"""Samples system metrics during the block, if the tracker can."""
    sample = getattr(tracking, 'sample_system_metrics', None)
    if sample is None or _rank() != 0:
        yield
        return
    with sample():
        yield


def record_failure(record: RunRecord) -> Optional[Exception]:
    r"""The error a run record says ended its run (its ``kumo.error``)."""
    error = record.tags.get('kumo.error')
    return None if error is None else RuntimeError(error)


def track_prediction(
        manifest: Optional['PredictionManifest'], tracking: Any,
        error: Optional[BaseException] = None,
        model: Any = None,
        summaries: Callable[[], Sequence['Summary']] = tuple
) -> Optional[Exception]:
    r"""Tracks a prediction run; returns the failure instead of raising."""
    if getattr(tracking, 'log_prediction_run', None) is None:
        return None
    try:
        if manifest is None:
            raise ValueError('the run has no prediction manifest')
        record = prediction_record(manifest, error, model, summaries())
    except Exception as failure:
        logger.warning('Experiment tracking failed: %s; the prediction run '
                       'is not tracked', full_error(failure), exc_info=_debug())
        return failure
    return log_run(tracking, 'log_prediction_run', record)


class SingleRunHooks:
    r"""Each prediction run and monitoring call as one run of its own."""

    def log_prediction_run(self, record: RunRecord) -> None:
        self.start(record)  # type: ignore[attr-defined]
        self.finish(record,  # type: ignore[attr-defined]
                    record_failure(record))

    log_monitoring = log_prediction_run


class Tracking(Protocol):
    r"""Records one `Trainer.fit` in an experiment tracker.

    Calls come from one background thread. `start` gets the fit run's
    record without metrics, `on_trial` (in a search of several trials) a
    trial attempt's record so far with the event that updated it, and `finish` the fit run's final record
    (`None` if tracking stopped early, which should end the run as failed).
    Optional members: ``prepare()``, called first on the fit's own thread;
    ``log_metrics(metrics, step)``, series of the fit run as they come (a
    single trial's epochs, or a search's best validation metric after each
    trial); ``tags`` (extra fit-run tags); ``log_model(bundle_dir, manifest,
    snapshot, registry)``, called after a fit that succeeded and before
    `finish`, which the fit waits for while the tracking thread is alive, up
    to an hour (an error, a stopped thread or the time limit ends the run as
    failed), and which returns the version it registered at ``registry``, a
    ``ModelRegistryLocation``; ``connector``, a connector whose model
    registry ``registry`` registers in when there is no ``log_model``; and
    ``platform``, ``run_id``, ``run_url``, ``experiment`` and
    ``logged_run_name``, the tracker's own run once `start` returned,
    recorded as the model manifest's ``tracking`` (`tracking_ref`); and
    the best-effort hooks ``log_prediction_run(record)``, whose system
    metrics a ``sample_system_metrics()`` context samples while the run
    works, and ``log_monitoring(record)``.
    """
    def start(self, record: RunRecord) -> None:
        ...

    def on_trial(self, record: RunRecord, event: Event) -> None:
        ...

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        ...


@dataclass
class _Search:
    r"""The fit run's series: one trial's epochs, or a search's best."""

    single: bool
    metric: str
    lower: bool
    best: Optional[Tuple[float, int]] = None

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> '_Search':
        from kumo_core.evaluation.suite import MetricStore

        metric = f"val_{params.get('training_job.tune_metric', 'loss')}"
        return cls(single=params.get('training_job.num_experiments') == 1,
                   metric=metric,
                   lower=MetricStore.tune_mode(metric) == 'min')

    def best_epoch(self, events: Sequence[Event]) -> Optional[int]:
        r"""The first epoch with the best validation metric."""
        points = [(event.epoch, event.metrics[self.metric])
                  for event in events if event.name == 'epoch_ended'
                  and event.epoch and self.metric in event.metrics]
        if not points:
            return None
        best = (min if self.lower else max)(value for _, value in points)
        return next(epoch for epoch, value in points if value == best)

    def on(self, event: Event) -> Optional[Tuple[Dict[str, float], int]]:
        if event.name == 'epoch_ended' and self.single:
            return dict(event.metrics), event.epoch or 0
        value = event.metrics.get(self.metric)
        if (event.name != 'trial_ended' or self.single or value is None
                or event.params.get('status') != 'succeeded'):
            return None
        if self.best is None or (value < self.best[0] if self.lower else
                                 value > self.best[0]):
            self.best = (value, event.trial or 0)
        return None


class RegistrationError(Exception):
    r"""Registering the logged model failed."""


class TrackingSession:
    def __init__(self, tracking: Optional[Tracking],
                 warn: bool = True) -> None:
        self._tracker = tracking if _rank() == 0 else None
        self._warn = warn
        self.spec: Optional[RunSpec] = None
        self._trials: Dict[int, List[Event]] = {}
        self._first_epoch: Dict[Any, Any] = {}
        self._search: Optional[_Search] = None
        self._jobs: 'queue.Queue[Optional[Callable[[], Any]]]' = queue.Queue(
            QUEUE_SIZE)
        self._worker: Optional[threading.Thread] = None
        self._behind = False
        self._started = threading.Event()
        self.error: Optional[BaseException] = None
        self.final: Optional['TrackingRef'] = None

    @property
    def enabled(self) -> bool:
        return self._tracker is not None

    def start(self, spec: Callable[[], RunSpec]) -> None:
        if self._tracker is None:
            return
        try:
            value = spec()
        except Exception as error:
            logger.warning('Experiment tracking failed: %s; the fit continues',
                           full_error(error), exc_info=_debug())
            self._fail(error)
            self._tracker = None
            return
        self.spec = value
        self._search = _Search.of(value.params)
        self._call(lambda tracker: getattr(tracker, 'prepare', _none)())
        self._worker = threading.Thread(target=self._work,
                                        name='kumo-tracking', daemon=True)
        self._worker.start()

        def begin(tracker: Tracking) -> None:
            tracker.start(run_record(dataclasses.replace(value, tags={
                **value.tags, **(getattr(tracker, 'tags', None) or {})
            })))

        started = self._submit(lambda: self._call(begin))
        if not (started and self._submit(self._started.set)):
            self._started.set()

    def pointer(self) -> Optional['TrackingRef']:
        r"""The tracker's own run, once it started (`tracking_ref`)."""
        if self.spec is None or not self._started.wait(FINISH_SECONDS):
            return None
        return tracking_ref(self._tracker)

    def on_event(self, event: Event) -> None:
        if self.spec is None or event.trial is None:
            return
        if event.name == 'trial_started':
            self._trials[event.trial] = []
        events = self._trials.setdefault(event.trial, [])
        if 'attempt' not in event.params and events:
            event = dataclasses.replace(event, params={
                **event.params, 'attempt': events[0].params['attempt']
            })
        events.append(event)
        if self._search is not None and not self._search.single:
            trial = trial_spec(events)
            record = run_record(dataclasses.replace(
                trial, model_label=self.spec.model_label, parents={
                    **{name: self.spec.parents.get(name)
                       for name in _TRIAL_PARENTS},
                    **trial.parents
                }))
            logged = dataclasses.replace(event,
                                         metrics=logged_names(event.metrics))
            for each in self._once_a_line(
                    (event.trial, event.params.get('attempt')), event,
                    logged):
                self._submit(lambda each=each: self._call(
                    lambda tracker: tracker.on_trial(record, each)))
        series = None if self._search is None else self._search.on(event)
        if series is not None:
            for metrics, step in self._once_a_line(
                    None, event, (logged_names(series[0]), series[1])):
                self._submit(lambda metrics=metrics, step=step: self._call(
                    lambda tracker: getattr(tracker, 'log_metrics', _none)(
                        metrics, step)))

    def _once_a_line(self, key: Any, event: Event,
                     item: Any) -> List[Any]:
        r"""An epoch held until its series' second, then both."""
        if event.name != 'epoch_ended':
            return [item]
        if key not in self._first_epoch:
            self._first_epoch[key] = item
            return []
        first, self._first_epoch[key] = self._first_epoch[key], None
        return [item] if first is None else [first, item]

    def log_model(
        self,
        bundle_dir: Path,
        manifest: Optional['ModelManifest'],
        snapshot: Any,
        registry: Optional['ModelRegistryLocation'] = None,
    ) -> Tuple[Optional['PublishedModel'], Optional[BaseException]]:
        r"""The tracker's ``log_model``, run on the tracking thread."""
        if self._worker is None or (self._tracker is not None and not callable(
                getattr(self._tracker, 'log_model', None))):
            return None, None
        done = threading.Event()
        failed: List[BaseException] = []
        registered: List[Any] = []

        def log() -> None:
            try:
                tracker = self._tracker
                if tracker is None:
                    raise RuntimeError(
                        'tracking stopped after an earlier failure' + (
                            '' if self.error is None else
                            f' ({full_error(self.error)})'))
                registered.append(
                    getattr(tracker, 'log_model')(bundle_dir, manifest,
                                                  snapshot, registry))
            except BaseException as error:
                failed.append(error)
            finally:
                done.set()

        worker = self._worker
        deadline = time.monotonic() + MODEL_SECONDS
        try:
            self._jobs.put(log, timeout=MODEL_SECONDS)
        except queue.Full:
            pass
        while not done.wait(1.0):
            if worker.is_alive() and time.monotonic() < deadline:
                continue
            failed.append(
                TimeoutError(f'it did not finish within {MODEL_SECONDS:.0f}s')
                if worker.is_alive() else
                RuntimeError('the tracking thread stopped'))
            tracker, self._tracker = self._tracker, None
            if tracker is not None:
                _abandon(tracker)
            break
        if failed:
            error = failed[0]
            verb = 'log'
            if isinstance(error, RegistrationError) and error.__cause__:
                verb, error = 'register', error.__cause__
            if self._warn:
                warn_failure(
                    f'Could not {verb} the model: {full_error(error)}',
                    error, 3, also=self.error)
            self._fail(error)
            return None, error
        return (registered or [None])[0], None

    def finish(self, spec: Optional[Callable[[], RunSpec]],
               error: Optional[BaseException]) -> None:
        r"""Ends the run, failed if error."""
        worker = self._worker
        if self.spec is None or worker is None:
            return
        try:
            final = self.spec if spec is None else spec()
            selected = None if self._search is None else self._search.best
            if error is None and selected is not None:
                final = dataclasses.replace(
                    final, steps=trial_spec(self._trials[selected[1]]).steps,
                    parents={**final.parents, 'best_trial': selected[1]})
            search = self._search
            if error is None and search is not None and (
                    selected or len(self._trials) == 1):
                kept = selected[1] if selected else next(iter(self._trials))
                epoch = search.best_epoch(self._trials[kept])
                if epoch is not None:
                    final = dataclasses.replace(
                        final, parents={**final.parents, 'best_epoch': epoch})
            held = self._first_epoch.get(None)
            final = _one_epoch_note(final, None if held is None else held[0])
            if error is not None and final.error is None:
                final = dataclasses.replace(final, error=format_error(error))
            record: Optional[RunRecord] = run_record(final)
        except Exception as failure:
            logger.warning('Could not build the tracking record: %s',
                           full_error(failure), exc_info=_debug())
            self._fail(failure)
            record = None

        def finish_tracker() -> None:
            tracker, self._tracker = self._tracker, None
            if tracker is not None:
                try:
                    tracker.finish(record, error)
                    self.final = tracking_ref(tracker)
                except Exception as failure:
                    self._fail(failure)
                    raise

        deadline = time.monotonic() + FINISH_SECONDS
        try:
            for job in (finish_tracker, None):
                self._jobs.put(job, timeout=max(0.0,
                                                deadline - time.monotonic()))
        except queue.Full:
            pass
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            self._fail(TimeoutError(
                f'experiment tracking did not finish within '
                f'{FINISH_SECONDS:g}s; it goes on in the background'))

    def _submit(self, job: Callable[[], Any]) -> bool:
        if self._behind:
            return False
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            self._behind = True
            logger.warning('Experiment tracking is falling behind; the rest '
                           'of this fit is not tracked')
            return False

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                job()
            except Exception as error:
                logger.warning('Experiment tracking failed: %s',
                               full_error(error), exc_info=_debug())

    def _call(self, call: Callable[[Tracking], Any]) -> None:
        tracker = self._tracker
        if tracker is None:
            return
        try:
            call(tracker)
        except Exception as error:
            logger.warning(
                'Experiment tracking failed: %s; it is off for the rest of '
                'this run', full_error(error), exc_info=_debug())
            self._fail(error)
            self._tracker = None
            _abandon(tracker)

    def _fail(self, error: BaseException) -> None:
        if self.error is None:
            self.error = error


def tracking_ref(tracker: Any) -> Optional['TrackingRef']:
    r"""The tracker's latest run."""
    from kumo.engine._manifest import TrackingRef

    run_id = getattr(tracker, 'run_id', None)
    if tracker is None or not run_id:
        return None
    return TrackingRef(
        platform=str(getattr(tracker, 'platform', type(tracker).__name__)),
        run_id=str(run_id), url=getattr(tracker, 'run_url', None),
        experiment=getattr(tracker, 'experiment', None),
        run_name=getattr(tracker, 'logged_run_name', None))


def log_run(tracking: Any, hook: str,
            record: RunRecord) -> Optional[Exception]:
    r"""Runs tracking.<hook>(record) on a thread; returns the failure."""
    method = getattr(tracking, hook, None)
    if method is None or _rank() != 0:
        return None
    failed: List[Exception] = []

    def log() -> None:
        try:
            method(record)
        except Exception as error:
            logger.warning('Experiment tracking failed: %s; the %s run is not '
                           'tracked', full_error(error),
                           record.tags.get('kumo.kind'), exc_info=_debug())
            failed.append(error)

    try:
        getattr(tracking, 'prepare', _none)()
    except Exception as error:
        logger.warning('Experiment tracking failed: %s; the %s run is not '
                       'tracked', full_error(error),
                       record.tags.get('kumo.kind'), exc_info=_debug())
        return error
    worker = threading.Thread(target=log, name='kumo-tracking', daemon=True)
    worker.start()
    worker.join(FINISH_SECONDS)
    if worker.is_alive():
        return TimeoutError(
            f'experiment tracking did not finish within {FINISH_SECONDS:g}s; '
            f'it goes on in the background')
    return failed[0] if failed else None


def _debug() -> bool:
    return logger.isEnabledFor(logging.DEBUG)


def _none(*args: Any) -> None:
    return None


def _abandon(tracker: Tracking) -> None:
    try:
        tracker.finish(None, None)
    except Exception:
        logger.debug('Could not end the run of %r', tracker, exc_info=True)


def _rank() -> int:
    for name in ('RANK', 'LOCAL_RANK'):
        if os.environ.get(name, '').strip() not in ('', '0'):
            return 1
    return 0


def _flatten(value: Any, prefix: str = '') -> Iterator[Tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _flatten(item, f'{prefix}{key}.')
    else:
        yield prefix[:-1], value


def _param_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)
