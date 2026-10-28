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
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from kumo_core.instrumentation import format_error

from kumo.engine._events import Event, _finite_floats
from kumo.engine._mlops import full_error, warn_failure

if TYPE_CHECKING:
    from kumoapi.plugin import ModelRegistryLocation, PublishedModel

    from kumo.engine._manifest import ModelManifest

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
    histories, ``artifacts`` map file names to text and ``inputs`` map
    source table names to their ``connector``, ``location``, ``version`` and
    ``version_kind``."""

    name: Optional[str] = None
    params: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    steps: Mapping[str, Tuple[Tuple[int, float],
                              ...]] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, Mapping[str, Optional[str]]] = field(
        default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    r"""What a run records, whatever trained or scored it: ``params`` are
    flat, ``parents`` become ``kumo.<name>`` pointer tags and ``tags`` are
    the user's own."""

    kind: str
    run_id: str
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
    status: Optional[str] = None
    error: Optional[str] = None


INPUT_FIELDS = ('connector', 'location', 'version', 'version_kind')


def fit_inputs(manifest: Any) -> Dict[str, Dict[str, Optional[str]]]:
    r"""Source tables of a training table or model manifest, read from its
    embedded parents; stub parents add none."""
    try:
        table = manifest
        if getattr(manifest, 'kind', None) == 'model':
            table = getattr(_parents(manifest), 'training_table', None)
        parents = _parents(table)
        snapshot = _parents(getattr(parents, 'snapshot', None))
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


def run_record(spec: RunSpec) -> RunRecord:
    params = {key: _param_value(value) for key, value in spec.params.items()}
    return RunRecord(
        name=spec.name,
        params=params,
        metrics=_finite_floats(spec.metrics),
        steps={name: tuple(history)
               for name, history in spec.steps.items()},
        tags={
            **_user_tags(spec.tags, params),
            'kumo.kind': spec.kind,
            'kumo.run_id': spec.run_id,
            **{f'kumo.{name}': str(value)
               for name, value in spec.parents.items() if value},
            **({} if spec.status is None else {'kumo.status': spec.status}),
            **({} if spec.error is None else {'kumo.error': spec.error}),
        },
        artifacts=dict(spec.artifacts),
        inputs=dict(spec.inputs),
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


def trial_spec(run_id: str, events: Sequence[Event]) -> RunSpec:
    r"""One trial attempt of run ``run_id``, from its events so far."""
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
        run_id=f'{run_id}-trial-{trial}-attempt-{attempt}',
        name=f'trial-{trial}-attempt-{attempt}',
        params=dict(started.params),
        metrics=metrics,
        steps=steps,
        parents={'parent_run': run_id},
        **ended,
    )


class Tracking(Protocol):
    r"""Records one `Trainer.fit` in an experiment tracker.

    Calls come from one background thread. `start` gets the fit run's
    record without metrics, `on_trial` a trial attempt's record so far with
    the event that updated it, and `finish` the fit run's final record
    (`None` if tracking stopped early, which should end the run as failed).
    Optional members: ``prepare()``, called first on the fit's own thread;
    ``tags`` (extra fit-run tags); ``log_model(bundle_dir, manifest,
    snapshot, registry)``, called after a fit that succeeded and before
    `finish`, which the fit waits for while the tracking thread is alive, up
    to an hour (an error, a stopped thread or the time limit ends the run as
    failed), and which returns the version it registered at ``registry``, a
    ``ModelRegistryLocation``.
    """
    def start(self, record: RunRecord) -> None:
        ...

    def on_trial(self, record: RunRecord, event: Event) -> None:
        ...

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        ...


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
        self._jobs: 'queue.Queue[Optional[Callable[[], Any]]]' = queue.Queue(
            QUEUE_SIZE)
        self._worker: Optional[threading.Thread] = None
        self._behind = False
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
        self._call(lambda tracker: getattr(tracker, 'prepare', _none)())
        self._worker = threading.Thread(target=self._work,
                                        name='kumo-tracking', daemon=True)
        self._worker.start()
        self._submit(lambda: self._call(lambda tracker: tracker.start(
            run_record(dataclasses.replace(value, tags={
                **value.tags, **(getattr(tracker, 'tags', None) or {})
            })))))

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
        record = run_record(trial_spec(self.spec.run_id, events))
        self._submit(lambda: self._call(
            lambda tracker: tracker.on_trial(record, event)))

    def log_model(
        self,
        bundle_dir: Path,
        manifest: Optional['ModelManifest'],
        snapshot: Any,
        registry: Optional['ModelRegistryLocation'] = None,
    ) -> Tuple[Optional['PublishedModel'], Optional[BaseException]]:
        r"""Runs the tracker's ``log_model`` on the tracking thread and waits
        for it; returns the registered version and the error it raised."""
        if self._worker is None or (self._tracker is not None and not hasattr(
                self._tracker, 'log_model')):
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
            logger.warning(
                'Experiment tracking did not finish within %ss; it goes on '
                'in the background and the fit returns', FINISH_SECONDS)

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
