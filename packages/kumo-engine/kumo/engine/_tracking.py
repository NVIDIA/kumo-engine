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
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from kumo_core.instrumentation import format_error
from kumo_core.summaries import Dataset, Scalar, Summary, Table, Tag

from kumo.engine._events import Event, _finite_floats
from kumo.engine._mlops import full_error, warn_failure

if TYPE_CHECKING:
    from kumoapi.plugin import ModelRegistryLocation, PublishedModel

    from kumo.engine._manifest import ModelManifest, TrackingRef

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
    summaries: Tuple[Summary, ...] = ()


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
    parents: Mapping[str, Optional[str]] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    summaries: Sequence[Summary] = ()
    status: Optional[str] = None
    error: Optional[str] = None


def logged_name(name: str) -> str:
    r"""``val_auroc`` as ``val/auroc``."""
    best = name.startswith('best_')
    split, _, metric = name.removeprefix('best_').partition('_')
    if split not in ('train', 'val', 'test') or not metric:
        return name
    return f"{split}/{'best_' if best else ''}{metric}"


def logged_names(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {logged_name(name): value for name, value in metrics.items()}


def by_step(steps: Mapping[str, Sequence[Tuple[int, float]]]
            ) -> List[Tuple[int, Dict[str, float]]]:
    grouped: Dict[int, Dict[str, float]] = {}
    for name, history in steps.items():
        for step, value in history:
            grouped.setdefault(step, {})[name] = value
    return sorted(grouped.items())


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
            **{f'kumo.{name}': str(value)
               for name, value in spec.parents.items() if value},
            **({} if spec.status is None else {'kumo.status': spec.status}),
            **({} if spec.error is None else {'kumo.error': spec.error}),
        },
        artifacts=dict(spec.artifacts),
        summaries=tuple(spec.summaries),
    )


def _user_tags(tags: Mapping[str, Any],
               params: Mapping[str, str]) -> Dict[str, str]:
    dropped = sorted(key for key in tags
                     if key.startswith('kumo.') or key in params)
    if dropped:
        warnings.warn(
            f'Tracking tags {dropped} are reserved and not recorded.',
            UserWarning, stacklevel=3)
    return {key: str(value) for key, value in tags.items()
            if key not in dropped}


def trial_spec(events: Sequence[Event]) -> RunSpec:
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


class Tracking(Protocol):
    r"""Records one `Trainer.fit` in an experiment tracker."""
    def start(self, record: RunRecord) -> None:
        ...

    def on_trial(self, record: RunRecord, event: Event) -> None:
        ...

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        ...


class TrackingBase(Tracking):
    r"""`Tracking` with defaults for its optional members."""

    run_id: Optional[str] = None
    run_url: Optional[str] = None
    tags: Optional[Mapping[str, str]] = None
    connector: Any = None
    log_model: Optional[Callable[..., Optional['PublishedModel']]] = None
    register: Optional[Callable[..., 'PublishedModel']] = None
    load_model: Optional[Callable[..., Any]] = None

    @property
    def platform(self) -> str:
        return type(self).__name__

    def prepare(self) -> None:
        pass

    def check(self, registry: Optional['ModelRegistryLocation'] = None
              ) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        pass

    def log_summaries(self, summaries: Sequence[Summary]) -> None:
        for summary in summaries:
            if isinstance(summary, Scalar):
                self._log_scalar(summary)
            elif isinstance(summary, Tag):
                self._log_tag(summary)
            elif isinstance(summary, Dataset):
                self._log_dataset(summary)
            else:
                self._log_table(summary)

    def _log_scalar(self, summary: Scalar) -> None:
        raise NotImplementedError

    def _log_tag(self, summary: Tag) -> None:
        raise NotImplementedError

    def _log_table(self, summary: Table) -> None:
        raise NotImplementedError

    def _log_dataset(self, summary: Dataset) -> None:
        pass


def table_json(table: Table) -> str:
    return json.dumps({'columns': list(table.columns),
                       'data': [list(row) for row in table.rows]}, default=str)


class _Foreign(TrackingBase):
    r"""A tracker that does not subclass `TrackingBase`, with its defaults."""

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker

    def __getattribute__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        if name in ('start', 'on_trial', 'finish'):
            return getattr(self._tracker, name)
        value = getattr(self._tracker, name, _MISSING)
        called = name in ('check', 'log_model', 'register')
        if value is _MISSING or called and not callable(value):
            return object.__getattribute__(self, name)
        return value

    @property
    def platform(self) -> str:
        return type(self._tracker).__name__

    def __repr__(self) -> str:
        return repr(self._tracker)


def as_tracking(tracking: Any) -> Optional[TrackingBase]:
    if tracking is None or isinstance(tracking, TrackingBase):
        return tracking
    return _Foreign(tracking)


@dataclass
class _Search:
    r"""The fit run's series: one trial's epochs, or a search's best."""

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
    r"""Registering, not logging, the model failed."""


class TrackingSession:
    def __init__(self, tracking: Any, warn: bool = True) -> None:
        self._tracker = as_tracking(tracking) if _rank() == 0 else None
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
        self._call(lambda tracker: tracker.prepare())
        self._worker = threading.Thread(target=self._work,
                                        name='kumo-tracking', daemon=True)
        self._worker.start()

        def begin(tracker: TrackingBase) -> None:
            tracker.start(run_record(dataclasses.replace(value, tags={
                **value.tags, **(tracker.tags or {})
            })))

        started = self._submit(lambda: self._call(begin))
        if not (started and self._submit(self._started.set)):
            self._started.set()

    def pointer(self) -> Optional['TrackingRef']:
        r"""The tracker's own run, once it started."""
        from kumo.engine._manifest import TrackingRef

        if self.spec is None or not self._started.wait(FINISH_SECONDS):
            return None
        tracker = self._tracker
        if tracker is None or not tracker.run_id:
            return None
        return TrackingRef(platform=str(tracker.platform),
                           run_id=str(tracker.run_id), url=tracker.run_url)

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
        record = run_record(trial_spec(events))
        logged = dataclasses.replace(event,
                                     metrics=logged_names(event.metrics))
        self._submit(lambda: self._call(
            lambda tracker: tracker.on_trial(record, logged)))
        series = None if self._search is None else self._search.on(event)
        if series is not None:
            metrics, step = logged_names(series[0]), series[1]
            self._submit(lambda: self._call(
                lambda tracker: tracker.log_metrics(metrics, step)))

    def log_model(
        self,
        bundle_dir: Path,
        manifest: Optional['ModelManifest'],
        snapshot: Any,
        registry: Optional['ModelRegistryLocation'] = None,
    ) -> Tuple[Optional['PublishedModel'], Optional[BaseException]]:
        if self._worker is None or (self._tracker is not None
                                    and self._tracker.log_model is None):
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
                registered.append(tracker.log_model(bundle_dir, manifest,
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
        r"""Ends the run with ``spec()`` (`None`: the started spec)."""
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

    def _call(self, call: Callable[[TrackingBase], Any]) -> None:
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


def _debug() -> bool:
    return logger.isEnabledFor(logging.DEBUG)


_MISSING = object()


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


def _param_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)
