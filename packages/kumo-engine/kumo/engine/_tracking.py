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

from kumo.engine._events import Event, _finite_floats

if TYPE_CHECKING:
    from kumo.engine.trainer.trainer import TrainingResult

logger = logging.getLogger(__name__)

DROPPED_PARAMS = 'kumo.dropped_params'
DROPPED_METRICS = 'kumo.dropped_metrics'
# Tracker calls run on one background thread so a slow tracking server
# never stalls training; these bound how long the fit waits for it.
QUEUE_SIZE = 1000
FINISH_SECONDS = 30.0


@dataclass(frozen=True)
class FitContext:
    r"""What a fit trains, known once its model plan is resolved."""

    run_id: str
    query: str
    model_plan: Mapping[str, Any]
    inputs: Mapping[str, Mapping[str, Optional[str]]] = field(
        default_factory=dict)


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


INPUT_FIELDS = ('connector', 'location', 'version', 'version_kind')


def fit_inputs(snapshot_dir: Path,
               table_dir: Path) -> Dict[str, Dict[str, Optional[str]]]:
    r"""Source tables recorded in the snapshot's manifest, plus a custom
    training table's source, if those manifests exist."""
    try:
        tables = dict(
            _parents(snapshot_dir / 'snapshot-manifest.json').get(
                'source_tables') or {})
        custom = _parents(table_dir / 'training-table-manifest.json').get(
            'source_table')
        if custom:
            tables['training_table'] = custom
        return {
            str(name): {key: _text(entry.get(key))
                        for key in INPUT_FIELDS}
            for name, entry in tables.items() if isinstance(entry, Mapping)
        }
    except Exception:
        logger.warning('Could not read the fit\'s source tables',
                       exc_info=True)
        return {}


def _parents(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8')).get('parents') or {}


def _text(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def run_record(context: FitContext, result: Optional['TrainingResult'] = None,
               fit_ended: Optional[Event] = None, *,
               tags: Optional[Mapping[str, str]] = None) -> RunRecord:
    r"""The fit run's record; ``tags`` are the user's extra tags."""
    metrics: Dict[str, float] = {}
    artifacts = {
        'query.pql': context.query,
        'model_plan.json': json.dumps(dict(context.model_plan), indent=2,
                                      default=str),
    }
    if result is not None:
        metrics.update(result.metrics)
        if fit_ended is not None and 'seconds' in fit_ended.metrics:
            metrics['seconds'] = fit_ended.metrics['seconds']
        provenance = Path(result.bundle_dir) / 'provenance.json'
        if provenance.is_file():
            artifacts['provenance.json'] = provenance.read_text(
                encoding='utf-8')
    params = {
        key: _param_value(value)
        for key, value in _flatten(context.model_plan)
    }
    return RunRecord(
        params=params,
        metrics=_finite_floats(metrics),
        tags={
            **_user_tags(tags, params),
            'kumo.kind': 'fit',
            'kumo.run_id': context.run_id,
            **_error_tag(fit_ended),
        },
        artifacts=artifacts,
        inputs=dict(context.inputs),
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


def _error_tag(ended: Optional[Event]) -> Dict[str, str]:
    if ended is None or 'error' not in ended.params:
        return {}
    return {'kumo.error': str(ended.params['error'])}


def trial_record(context: FitContext, events: Sequence[Event]) -> RunRecord:
    r"""One trial attempt's record from its events so far."""
    started = next(event for event in events if event.name == 'trial_started')
    trial, attempt = started.trial, int(started.params['attempt'])
    params: Dict[str, str] = {
        key: _param_value(value)
        for key, value in started.params.items()
    }
    metrics: Dict[str, float] = {}
    steps: Dict[str, List[Tuple[int, float]]] = {}
    ended: Dict[str, str] = {}
    for event in events:
        if event.name == 'epoch_ended' and event.epoch is not None:
            for name, value in event.metrics.items():
                steps.setdefault(name, []).append((event.epoch, value))
        elif event.name == 'trial_ended':
            ended = {
                'kumo.status': str(event.params.get('status')),
                **_error_tag(event),
            }
            metrics.update({
                name if name == 'seconds' else f'best_{name}': value
                for name, value in event.metrics.items()
            })
    return RunRecord(
        name=f'trial-{trial}-attempt-{attempt}',
        params=params,
        metrics=metrics,
        steps={
            name: tuple(history)
            for name, history in steps.items()
        },
        tags={
            'kumo.kind': 'trial',
            'kumo.run_id': f'{context.run_id}-trial-{trial}-attempt-{attempt}',
            'kumo.parent_run': context.run_id,
            **ended,
        },
    )


class Tracking(Protocol):
    r"""Records one `Trainer.fit` in an experiment tracker.

    Calls come from one background thread. `start` gets the fit run's
    record without metrics, `on_trial` a trial attempt's record so far with
    the event that updated it, and `finish` the fit run's final record
    (`None` if tracking stopped early, which should end the run as failed).
    Optional members: ``prepare()``, called first on the fit's own thread,
    and ``tags`` (extra fit-run tags).
    """
    def start(self, record: RunRecord) -> None:
        ...

    def on_trial(self, record: RunRecord, event: Event) -> None:
        ...

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        ...


class TrackingSession:
    def __init__(self, tracking: Optional[Tracking]) -> None:
        self._tracker = tracking if _rank() == 0 else None
        self._context: Optional[FitContext] = None
        self._trials: Dict[int, List[Event]] = {}
        self._fit_ended: Optional[Event] = None
        self._jobs: 'queue.Queue[Optional[Callable[[], Any]]]' = queue.Queue(
            QUEUE_SIZE)
        self._worker: Optional[threading.Thread] = None
        self._behind = False

    @property
    def enabled(self) -> bool:
        return self._tracker is not None

    def start(self, context: Callable[[], FitContext]) -> None:
        if self._tracker is None:
            return
        try:
            value = context()
        except Exception:
            logger.warning('Experiment tracking failed; the fit continues',
                           exc_info=True)
            self._tracker = None
            return
        self._context = value
        self._call(lambda tracker: getattr(tracker, 'prepare', _none)())
        self._worker = threading.Thread(target=self._work,
                                        name='kumo-tracking', daemon=True)
        self._worker.start()
        self._submit(lambda: self._call(lambda tracker: tracker.start(
            run_record(value, tags=getattr(tracker, 'tags', None)))))

    def on_event(self, event: Event) -> None:
        if self._context is None:
            return
        if event.name == 'fit_ended':
            self._fit_ended = event
        if event.trial is None:
            return
        if event.name == 'trial_started':
            self._trials[event.trial] = []
        events = self._trials.setdefault(event.trial, [])
        if 'attempt' not in event.params and events:
            event = dataclasses.replace(event, params={
                **event.params, 'attempt': events[0].params['attempt']
            })
        events.append(event)
        record = trial_record(self._context, events)
        self._submit(lambda: self._call(
            lambda tracker: tracker.on_trial(record, event)))

    def finish(self, result: Optional['TrainingResult'],
               error: Optional[BaseException]) -> None:
        worker = self._worker
        if self._context is None or worker is None:
            return
        try:
            record: Optional[RunRecord] = run_record(self._context, result,
                                                     self._fit_ended)
        except Exception:
            logger.warning('Could not build the tracking record',
                           exc_info=True)
            record = None

        def finish_tracker() -> None:
            tracker, self._tracker = self._tracker, None
            if tracker is not None:
                tracker.finish(record, error)

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
            except Exception:
                logger.warning('Experiment tracking failed', exc_info=True)

    def _call(self, call: Callable[[Tracking], Any]) -> None:
        tracker = self._tracker
        if tracker is None:
            return
        try:
            call(tracker)
        except Exception:
            logger.warning(
                'Experiment tracking failed; it is off for the rest of this '
                'run', exc_info=True)
            self._tracker = None
            _abandon(tracker)


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
