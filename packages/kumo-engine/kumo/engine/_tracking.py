# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Platform-neutral run records and the `Tracking` protocol."""

import dataclasses
import hashlib
import json
import logging
import os
import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
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
    Union,
)

from kumo.engine._events import Event
from kumo.engine._version import __version__

if TYPE_CHECKING:
    from kumo.engine.trainer.trainer import TrainingResult

logger = logging.getLogger(__name__)

# Lightning logs per global step; the prefix keeps those apart from the
# per-epoch ``steps``.
LIGHTNING_PREFIX = 'lightning'
DROPPED_PARAMS = 'kumo.dropped_params'
DROPPED_METRICS = 'kumo.dropped_metrics'


@dataclass(frozen=True)
class FitContext:
    r"""What a fit trains, known once its model plan is resolved."""

    run_id: str
    task_type: str
    snapshot_id: str
    training_table_id: str
    query: str
    model_plan: Mapping[str, Any]


@dataclass(frozen=True)
class RunRecord:
    r"""One tracked run; ``steps`` are per-epoch ``(epoch, value)``
    histories and ``artifacts`` map file names to text."""

    name: Optional[str] = None
    params: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    steps: Mapping[str, Tuple[Tuple[int, float],
                              ...]] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    links: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    r"""What a tracking adapter records beyond params, metrics and tags.

    Args:
        nested_runs: trial runs are child runs of the fit run.
        model_link: published models are also linked through the platform's
            own run-to-model link.
    """

    nested_runs: bool = False
    system_metrics: bool = False
    step_metrics: bool = False
    artifacts: bool = False
    model_link: bool = False


def run_record(context: FitContext, result: Optional['TrainingResult'] = None,
               fit_ended: Optional[Event] = None, *,
               tags: Optional[Mapping[str, str]] = None) -> RunRecord:
    r"""The fit run's record; metrics need `result` and the ``fit_ended``
    event, which also sets ``kumo.error`` when the fit failed.

    ``tags`` (an adapter's user-supplied tags) pass through under their given
    names, except ``kumo.*`` names, which only the engine sets, and names of
    the record's params (both dropped with a warning).
    """
    metrics: Dict[str, float] = {}
    links: Dict[str, str] = {}
    if result is not None:
        metrics.update(result.metrics)
        if fit_ended is not None and 'seconds' in fit_ended.metrics:
            metrics['seconds'] = fit_ended.metrics['seconds']
        if result.published_model is not None:
            links = {
                'model_name': result.published_model.name,
                'model_version': str(result.published_model.version),
                'model_uri': result.published_model.uri,
            }
    params = {
        'task_type': context.task_type,
        **{
            key: _param_value(value)
            for key, value in _flatten(context.model_plan)
        },
    }
    return RunRecord(
        params=params,
        metrics=metrics,
        tags={
            **_user_tags(tags, params),
            'kumo.kind': 'fit',
            'kumo.engine_version': __version__,
            'kumo.run_id': context.run_id,
            'kumo.snapshot_id': context.snapshot_id,
            'kumo.training_table_id': context.training_table_id,
            'kumo.query_sha256': hashlib.sha256(
                context.query.encode()).hexdigest(),
            **_error_tag(fit_ended),
            **{f'kumo.{key}': value
               for key, value in links.items()},
        },
        links=links,
        artifacts={
            'query.pql':
            context.query,
            'model_plan.json':
            json.dumps(dict(context.model_plan), indent=2, default=str),
        },
    )


def _user_tags(tags: Optional[Mapping[str, Any]],
               params: Mapping[str, str]) -> Dict[str, str]:
    dropped = sorted(key for key in tags or {}
                     if key.startswith('kumo.') or key in params)
    if dropped:
        warnings.warn(
            f'Tracking tags {dropped} are not recorded: kumo.* tags are set '
            f'by the engine, and a tag may not share a param\'s name.',
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
    r"""One trial attempt's record from its events so far: ``trial_started``,
    then ``epoch_ended`` and ``trial_ended`` (which sets ``kumo.status`` and,
    on failure, ``kumo.error``)."""
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

    Pass an implementation (or a sequence of them) as ``fit(tracking=...)``.
    Hooks receive platform-neutral facts; build what to log with `run_record`
    and `trial_record`. Only global rank 0 calls the hooks. A hook that raises
    is logged as a warning and its tracker is not called again for that fit;
    it never fails the fit or affects other trackers.

    Optional members:

    - ``trial_logger(trial, attempt)``: a Lightning logger that also receives
      that search-trial attempt's step metrics, or None. Lightning calls the
      returned logger directly, so its methods must not raise.
    - ``capabilities``: a `Capabilities` declaring what the adapter records.
    """
    def start(self, context: FitContext) -> None:
        r"""Called once the model plan is resolved."""
        ...

    def on_event(self, event: Event) -> None:
        r"""Called for every progress event after `start`. Trial events
        carry ``params['attempt']``."""
        ...

    def finish(self, result: Optional['TrainingResult'],
               error: Optional[BaseException]) -> None:
        r"""Called once when the fit returns or raises."""
        ...


Trackers = Union[Tracking, Sequence[Tracking], None]


class TrackingSession:
    r"""Calls one fit's trackers, each best-effort and isolated: a tracker
    whose hook raises is dropped for the rest of the fit."""
    def __init__(self, tracking: Trackers) -> None:
        if tracking is None or _rank() != 0:
            trackers: List[Tracking] = []
        elif isinstance(tracking, SequenceABC):
            trackers = list(tracking)
        else:
            trackers = [tracking]
        self._trackers = trackers
        self._started = False
        self._attempts: Dict[int, int] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._trackers)

    def start(self, context: Callable[[], FitContext]) -> None:
        if not self._trackers:
            return
        self._started = True
        try:
            value = context()
        except Exception:
            logger.warning('Experiment tracking failed; the fit continues',
                           exc_info=True)
            self._trackers = []
            return
        self._call_all('start', value)

    def trial_logger(self, trial: int, attempt: int) -> Optional[Any]:
        loggers = [
            pl_logger for pl_logger in self._call_all(
                'trial_logger', trial, attempt) if pl_logger is not None
        ]
        return loggers or None

    def on_event(self, event: Event) -> None:
        if not self._started:
            return
        try:
            event = _with_attempt(event, self._attempts)
        except Exception:
            logger.warning('Could not read the trial attempt of %r', event,
                           exc_info=True)
        self._call_all('on_event', event)

    def finish(self, result: Optional['TrainingResult'],
               error: Optional[BaseException]) -> None:
        if not self._started:
            return
        self._call_all('finish', result, error)

    def _call_all(self, hook: str, *args: Any) -> List[Any]:
        values = []
        for tracker in list(self._trackers):
            function = getattr(tracker, hook, None)
            if not callable(function):
                continue
            try:
                values.append(function(*args))
            except Exception:
                logger.warning(
                    'Experiment tracking failed in %r; it is off for the '
                    'rest of this fit, which continues', tracker,
                    exc_info=True)
                self._trackers.remove(tracker)
        return values


def _with_attempt(event: Event, attempts: Dict[int, int]) -> Event:
    if event.trial is None:
        return event
    if event.name == 'trial_started':
        attempts[event.trial] = int(event.params.get('attempt', 1))
    if 'attempt' in event.params:
        return event
    return dataclasses.replace(
        event, params={
            **event.params, 'attempt': attempts.get(event.trial, 1)
        })


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
