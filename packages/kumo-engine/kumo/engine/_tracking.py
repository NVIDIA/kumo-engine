# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Platform-neutral run records and the `Tracking` protocol."""

import dataclasses
import json
import logging
import os
import queue
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
# Tracker calls run on one background thread so a slow tracking server
# never stalls training; these bound how long the fit waits for it.
QUEUE_SIZE = 1000
FINISH_SECONDS = 30.0
MODEL_SECONDS = 3600.0


@dataclass(frozen=True)
class RunRecord:
    r"""One tracked run; ``steps`` are per-epoch ``(epoch, value)``
    histories, ``artifacts`` map file names to text, ``inputs`` map
    source table names to their ``connector``, ``location``, ``version`` and
    ``version_kind``, ``summaries`` are `kumo_core.summaries` results,
    which each tracker logs in its own way, and ``description`` is Markdown
    for trackers that describe runs."""

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


@dataclass(frozen=True)
class RunSpec:
    r"""What a run records, whatever trained or scored it: ``params`` are
    flat, ``parents`` become ``kumo.<name>`` pointer tags and ``tags`` are
    the user's own."""

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


INPUT_FIELDS = ('connector', 'location', 'version', 'version_kind')


def fit_inputs(manifest: Any) -> Dict[str, Dict[str, Optional[str]]]:
    r"""Source tables of a training table or model manifest, or a prediction
    run's scoring snapshot, read from its embedded parents; stub parents add
    none."""
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


def logged_name(name: str) -> str:
    r"""A metric's name in a tracker: ``val_auroc`` is ``val/auroc`` and a
    trial's ``best_val_auroc`` is ``val/best_auroc``, so charts group by
    split."""
    best = name.startswith('best_')
    split, _, metric = name.removeprefix('best_').partition('_')
    if split not in ('train', 'val') or not metric:
        return name
    return f"{split}/{'best_' if best else ''}{metric}"


def logged_names(metrics: Mapping[str, Any],
                 refit_full: bool = False) -> Dict[str, Any]:
    r"""`logged_name` of each metric, and the holdout's ``test_<m>`` as the
    shipped model's ``<m>`` (``test_loss`` as ``holdout_loss``). After a
    train-and-validation refit, its ``test_refit_<m>`` are, and the model
    before it is ``before_refit/<m>``. After a refit on all data (``refit_full``)
    no holdout describes the shipped model: the last model tested is
    ``before_refit/<m>``."""
    refit = any(name.startswith('test_refit_') for name in metrics)
    named = {}
    for name, value in metrics.items():
        if not name.startswith('test_'):
            named[logged_name(name)] = value
            continue
        metric = name.removeprefix('test_')
        latest = metric.startswith('refit_') == refit
        metric = metric.removeprefix('refit_')
        metric = 'holdout_loss' if metric == 'loss' else metric
        if not latest and refit_full:
            continue
        named[metric if latest and not refit_full else
              f'before_refit/{metric}'] = value
    return named


def by_step(steps: Mapping[str, Sequence[Tuple[int, float]]]
            ) -> List[Tuple[int, Dict[str, float]]]:
    r"""Per-step histories regrouped as each step's metrics, in step
    order."""
    grouped: Dict[int, Dict[str, float]] = {}
    for name, history in steps.items():
        for step, value in history:
            grouped.setdefault(step, {})[name] = value
    return sorted(grouped.items())


#: The fit's pointer tags its trial runs share.
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
                name if name == 'seconds' else f'best_{name}': value
                for name, value in event.metrics.items()
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
    r"""A prediction run's record, from its manifest: scoring options as
    params, row counts and score summaries (with drift against the holdout
    of ``model``, the `TrainingResult` that scored), the scoring
    snapshot's source tables as inputs, pointer tags (``kumo.error`` when
    ``error`` ended the run), and the manifest as an artifact."""
    from kumo.engine.trainer._provenance import holdout_counts

    table = manifest.prediction_table
    params = {**manifest.scoring.to_dict(), 'anchors': table.anchors}
    model_manifest = getattr(model, 'manifest', None)
    return run_record(
        RunSpec(
            kind='prediction',
            name=run_name('predict', model, _started(manifest)),
            params={
                key: value
                for key, value in params.items() if value is not None
            },
            parents={
                'bundle_id': manifest.parents.model.id,
                'prediction_id': manifest.id,
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
                manifest, holdout_counts(model_manifest)), *summaries),
            error=None if error is None else format_error(error),
            description=_unlabelled(getattr(model_manifest, 'tracking',
                                            None)),
        ))


def prediction_parents(manifest: Optional['PredictionManifest']
                       ) -> Mapping[str, Any]:
    r"""Pointer tags to the scoring snapshot and the task type of the
    prediction run ``manifest`` records."""
    if manifest is None:
        return {}
    training_table = getattr(_parents(manifest.parents.model),
                             'training_table', None)
    return {'snapshot_id': manifest.parents.snapshot.id,
            'task_type': getattr(training_table, 'task_type', None)}


def run_name(kind: str, model: Any, started: datetime) -> str:
    r"""``<kind>-<model>-v<version>-<YYYYMMDD-HHMM>`` (UTC) for a run
    started at ``started`` with ``model``, a `TrainingResult`, or
    ``<kind>-<YYYYMMDD-HHMM>`` if the model is not registered."""
    registered = getattr(model, 'registered_model', None)
    return '-'.join(filter(None, (
        kind, None if registered is None else
        f'{registered.name}-v{registered.version}',
        started.astimezone(timezone.utc).strftime('%Y%m%d-%H%M'))))


def fit_run_id(manifest: Any) -> Optional[str]:
    r"""The tracker's run of the fit a `ModelManifest` records."""
    tracking = getattr(manifest, 'tracking', None)
    return None if tracking is None else tracking.run_id


def registered_parents(model: Any) -> Mapping[str, Any]:
    r"""Pointer tags to the registered version of ``model``, a
    `TrainingResult`: ``model_name``, ``model_version`` and
    ``model_alias``."""
    from kumo.engine.trainer._run_spec import _registered

    if model is None:
        return {}
    return _registered(model.registered_model, model.registered_alias)


def _started(manifest: 'PredictionManifest') -> datetime:
    timings = manifest.timings
    return (datetime.now(timezone.utc) if timings is None else
            datetime.fromisoformat(timings.started_at))


def _unlabelled(fit: Optional['TrackingRef']) -> str:
    text = ('Scores of unlabelled rows: this run records their distribution '
            'and its drift from the holdout, not accuracy, which '
            '`Trainer.evaluate` logs once the outcomes are known.')
    if fit is None:
        return text
    link = f'[{fit.run_id}]({fit.url})' if fit.url else fit.run_id
    return f'{text} Fit run: {link}.'


def _prediction_summaries(manifest: 'PredictionManifest',
                          baseline: Any) -> List['Summary']:
    r"""``pred/rows``, ``pred/skipped_share`` and a note of the rows not
    scored; the scores' summaries from their exact counts, with their drift
    from ``baseline``, the holdout's."""
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
            baseline=baseline)
    return summaries


@contextmanager
def system_metrics(tracking: Any) -> Iterator[None]:
    r"""Has ``tracking`` sample system metrics during the block, through its
    optional ``sample_system_metrics()`` context, for the prediction or
    monitoring run it logs next."""
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
        summaries: Sequence['Summary'] = ()) -> Optional[Exception]:
    r"""Record a prediction run of ``model`` in ``tracking``, failed if
    ``error`` ended it, with ``summaries`` besides its scores'; returns the
    tracking failure instead of raising it."""
    if getattr(tracking, 'log_prediction_run', None) is None:
        return None
    try:
        if manifest is None:
            raise ValueError('the run has no prediction manifest')
        record = prediction_record(manifest, error, model, summaries)
    except Exception as failure:
        logger.warning('Experiment tracking failed: %s; the prediction run '
                       'is not tracked', full_error(failure), exc_info=_debug())
        return failure
    return log_run(tracking, 'log_prediction_run', record)


class Tracking(Protocol):
    r"""Records one `Trainer.fit` in an experiment tracker.

    Calls come from one background thread. `start` gets the fit run's
    record without metrics, `on_trial` a trial attempt's record so far with
    the event that updated it, and `finish` the fit run's final record
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
    ``platform``, ``run_id`` and ``run_url``, the tracker's own run once
    `start` returned, recorded as the model manifest's ``tracking``; and
    the best-effort hooks ``log_prediction_run(record)`` and
    ``log_monitoring(record)``, whose system metrics a
    ``sample_system_metrics()`` context samples while the run works.
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
    r"""The fit run's own series: a single trial's epochs as they end, or,
    for a search, the best validation ``metric`` after each trial; then
    `best`, the selected trial, whose epochs the run records at the end."""

    single: bool
    metric: str
    lower: bool
    best: Optional[Tuple[float, int]] = None
    trials: int = 0

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> '_Search':
        from kumo_core.evaluation.suite import MetricStore

        metric = f"val_{params.get('training_job.tune_metric', 'loss')}"
        return cls(single=params.get('training_job.num_experiments') == 1,
                   metric=metric,
                   lower=MetricStore.tune_mode(metric) == 'min')

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
        self.trials += 1
        return {f'search/best_{self.metric}': self.best[0]}, self.trials


class RegistrationError(Exception):
    r"""Raised by a tracker's ``log_model`` from the error that registering
    the logged model raised."""


class TrackingSession:
    def __init__(self, tracking: Optional[Tracking],
                 warn: bool = True) -> None:
        self._tracker = tracking if _rank() == 0 else None
        self._warn = warn
        self.spec: Optional[RunSpec] = None
        self._trials: Dict[int, List[Event]] = {}
        self._search: Optional[_Search] = None
        self._jobs: 'queue.Queue[Optional[Callable[[], Any]]]' = queue.Queue(
            QUEUE_SIZE)
        self._worker: Optional[threading.Thread] = None
        self._behind = False
        self._started = threading.Event()
        self.error: Optional[BaseException] = None

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
        r"""The tracker's own run, once it started: its platform, run id
        and URL."""
        from kumo.engine._manifest import TrackingRef

        if self.spec is None or not self._started.wait(FINISH_SECONDS):
            return None
        tracker = self._tracker
        run_id = getattr(tracker, 'run_id', None)
        if tracker is None or not run_id:
            return None
        return TrackingRef(
            platform=str(getattr(tracker, 'platform', type(tracker).__name__)),
            run_id=str(run_id), url=getattr(tracker, 'run_url', None))

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
        trial = trial_spec(events)
        record = run_record(dataclasses.replace(trial, parents={
            **{name: self.spec.parents.get(name) for name in _TRIAL_PARENTS},
            **trial.parents
        }))
        logged = dataclasses.replace(event,
                                     metrics=logged_names(event.metrics))
        self._submit(lambda: self._call(
            lambda tracker: tracker.on_trial(record, logged)))
        series = None if self._search is None else self._search.on(event)
        if series is not None:
            metrics, step = logged_names(series[0]), series[1]
            self._submit(lambda: self._call(
                lambda tracker: getattr(tracker, 'log_metrics', _none)(
                    metrics, step)))

    def log_model(
        self,
        bundle_dir: Path,
        manifest: Optional['ModelManifest'],
        snapshot: Any,
        registry: Optional['ModelRegistryLocation'] = None,
    ) -> Tuple[Optional['PublishedModel'], Optional[BaseException]]:
        r"""Runs the tracker's ``log_model`` on the tracking thread and waits
        for it; returns the registered version and the error it raised."""
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
        r"""Ends the run with ``spec()``, its final spec (`None`: the started
        one), failed if ``error``."""
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


def log_run(tracking: Any, hook: str,
            record: RunRecord) -> Optional[Exception]:
    r"""Log a standalone run with ``tracking.<hook>(record)`` on a tracking
    thread, waiting up to `FINISH_SECONDS`; returns the tracking failure
    instead of raising it."""
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
        logger.warning(
            'Experiment tracking did not finish within %ss; it goes on in '
            'the background', FINISH_SECONDS)
    return failed[0] if failed else None


def _debug() -> bool:
    r"""Tracebacks of tracking failures only at DEBUG: the failure itself is
    reported once, by the call that tracked it."""
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
