# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importable payload and observer executed inside a Snowflake ML Job."""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import threading
import time
import traceback
from datetime import date, datetime
from datetime import time as time_value
from datetime import timedelta, timezone
from decimal import Decimal
from enum import Enum
from functools import wraps
from pathlib import PurePath
from typing import Any, Callable, Mapping, Sequence

from kumo_core import runtime_progress
from kumoapi.common import JobStatus
from kumoapi.jobs import (
    AutoTrainerProgress,
    JobEventLogEntry,
    JobStatusReport,
    PredictionProgress,
)
from kumoapi.json_serde import to_json_dict
from kumoapi.plugin import PublishedModel

from ._ml_job_stage import StageStore
from .ml_job_contracts import (
    _TRANSPORT_KEY,
    _parse_transport_node,
    _published_model_from_payload,
    _published_model_payload,
    _transport_node,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_MIN_PUBLISH_INTERVAL_S = 1.0
_RESULT_PUBLISH_ATTEMPTS = 3
_RESULT_PUBLISH_RETRY_S = 0.1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _active_job_session() -> Any:
    """Resolve Snowflake's container-owned active session at execution time."""
    from snowflake.snowpark import Session
    return Session.builder.getOrCreate()


def _encoded_user_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if _TRANSPORT_KEY in value:
        return _transport_node('mapping', value)
    return dict(value)


def _json_value(value: Any, *, path: str = 'result') -> Any:
    """Convert the deliberately narrow public result contract to JSON."""
    if isinstance(value, PublishedModel):
        return _transport_node('published_model',
                               _published_model_payload(value))
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f'{path} contains a non-finite float')
        return value
    if isinstance(value, (datetime, date, time_value)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return _json_value(value.value, path=path)
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f'{path} has non-string mapping key {key!r}')
            converted[key] = _json_value(item, path=f'{path}.{key}')
        return _encoded_user_mapping(converted)
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f'{path}[{index}]')
            for index, item in enumerate(value)
        ]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encoded_user_mapping({
            field.name:
            _json_value(getattr(value, field.name), path=f'{path}.{field.name}')
            for field in dataclasses.fields(value)
        })

    raise TypeError(
        f'{path} has unsupported type {type(value).__name__}; return JSON '
        'values, dataclasses containing JSON values, paths, or a PublishedModel'
    )


def _metric_snapshots(
    progress: AutoTrainerProgress,
) -> tuple[dict[str, float], dict[str, float]]:
    observations: dict[str, list[float]] = {}
    latest: dict[str, float] = {}
    trials = list(progress.trial_progress.values())
    for trial in trials:
        for _, epoch in sorted(trial.metrics.items()):
            metrics = epoch.validation_metrics or epoch.train_metrics
            observed = {}
            for name, value in metrics.items():
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    observed[str(name)] = number
            for name, value in observed.items():
                observations.setdefault(name, []).append(value)
        if trial is trials[-1] and trial.metrics:
            final_epoch = trial.metrics[max(trial.metrics)]
            metrics = final_epoch.validation_metrics or final_epoch.train_metrics
            latest = {}
            for name, value in metrics.items():
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    latest[str(name)] = number
    best = {}
    for name, values in observations.items():
        lowered = name.lower()
        minimize = any(token in lowered
                       for token in ('loss', 'error', 'mae', 'mse', 'rmse'))
        best[name] = min(values) if minimize else max(values)
    return latest, best


def _observability_json(value: Any) -> Any:
    """Make native progress JSON-safe without affecting workload success."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                continue
            result[str(key)] = _observability_json(item)
        return result
    if isinstance(value, list):
        return [_observability_json(item) for item in value]
    return value


def _safe_exception_text(error: BaseException) -> str:
    """Format an exception without letting broken ``__str__`` mask it."""
    try:
        return str(error)
    except Exception:
        return f'<{type(error).__name__} message unavailable>'


def _safe_traceback(error: BaseException) -> str:
    """Best-effort traceback text for a durable failure envelope."""
    try:
        return ''.join(traceback.format_exception(error))
    except Exception:
        return f'{type(error).__name__}: {_safe_exception_text(error)}'


def _synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def synchronized(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized


class _RemoteObserver:
    """Serially mirrors native progress through the ML Job active session."""
    def __init__(self, store: StageStore, config: Mapping[str, Any]) -> None:
        self._lock = threading.RLock()
        self._store = store
        self._execution_id = str(config['execution_id'])
        self._progress_location = str(config['progress_location'])
        self._started_at = _now()
        self._updated_at = self._started_at
        self._events = [
            JobEventLogEntry(
                stage_name='Starting user function',
                last_updated_at=self._started_at,
            )
        ]
        self._operation_kind: str | None = None
        self._operation_id: str | None = None
        self._training: AutoTrainerProgress | None = None
        self._prediction: PredictionProgress | None = None
        self._current_metrics: dict[str, float] = {}
        self._best_metrics: dict[str, float] = {}
        self._operation_active = False
        self._operation_automatic = False
        self._active_automatic_source: Any | None = None
        self._completed_automatic_source: Any | None = None
        self._policy_error: RuntimeError | None = None
        self._last_publish = 0.0
        self._publish(force=True)

    def _payload(self, status: JobStatus = JobStatus.RUNNING) -> dict[str, Any]:
        report = JobStatusReport(
            status=status,
            tracking_url='',
            start_time=self._started_at,
            end_time=self._updated_at if status.is_terminal else None,
            event_log=list(self._events),
        )
        return _observability_json({
            'schema_version':
            _SCHEMA_VERSION,
            'execution_id':
            self._execution_id,
            'operation_kind':
            self._operation_kind,
            'operation_id':
            self._operation_id,
            'updated_at':
            self._updated_at.isoformat(),
            'status_report':
            to_json_dict(report),
            'training':
            (None if self._training is None else _observability_json(
                to_json_dict(self._training))),
            'prediction': (None if self._prediction is None else to_json_dict(
                self._prediction)),
            'current_metrics':
            dict(self._current_metrics),
            'best_metrics':
            dict(self._best_metrics),
        })

    def _publish(self, *, force: bool = False,
                 status: JobStatus = JobStatus.RUNNING) -> None:
        monotonic = time.monotonic()
        if not force and monotonic - self._last_publish < _MIN_PUBLISH_INTERVAL_S:
            return
        self._updated_at = _now()
        try:
            self._store.put_json(self._progress_location, self._payload(status))
        except Exception:
            # Progress is observability, not the user's workload result.  A
            # transient stage write must not abort an otherwise valid model
            # run or replace its exception.  The typed terminal result below
            # remains the authoritative completion transport.
            logger.warning(
                'Could not publish progress for Kumo execution %s',
                self._execution_id,
                exc_info=True,
            )
        self._last_publish = monotonic

    @_synchronized
    def operation_started(self, kind: str, operation_id: str,
                          stage: str) -> None:
        if self._operation_active:
            error = RuntimeError(
                'A submitted Snowflake ML Job may run only one top-level '
                'Kumo operation at a time; overlapping '
                f'{self._operation_kind!r} and {kind!r} is unsupported')
            self._policy_error = error
            raise error
        self._operation_kind = kind
        self._operation_id = operation_id
        self._training = None
        self._prediction = None
        self._current_metrics.clear()
        self._best_metrics.clear()
        self._operation_active = True
        self._active_automatic_source = None
        self._completed_automatic_source = None
        self.stage_changed(stage)

    @_synchronized
    def raise_for_policy_violation(self) -> None:
        """Fail the durable result even if callback delivery was best effort."""
        if self._policy_error is not None:
            raise self._policy_error

    @_synchronized
    def stage_changed(self, stage: str, detail: str | None = None) -> None:
        now = _now()
        if not self._events or self._events[-1].stage_name != stage:
            self._events.append(
                JobEventLogEntry(
                    stage_name=stage,
                    last_updated_at=now,
                    detail=detail,
                ))
        else:
            self._events[-1] = JobEventLogEntry(
                stage_name=stage,
                last_updated_at=now,
                detail=detail,
            )
        self._publish(force=True)

    @_synchronized
    def progress_updated(self, kind: str, progress: Any) -> None:
        # Per-trial callbacks cannot establish a top-level operation.
        if not isinstance(progress, (AutoTrainerProgress, PredictionProgress)):
            return
        if (not self._operation_active and kind == self._operation_kind
                and progress is self._completed_automatic_source):
            # All callbacks from one native logger carry the same mutable
            # progress object.  Retain that object as the automatic-operation
            # generation marker so a delayed callback cannot reopen it.  A new
            # logger object can start the next sequential operation.
            return
        if not self._operation_active:
            completed_source = self._completed_automatic_source
            stage = {
                'training': 'Training',
                'prediction': 'Predicting',
            }.get(kind, kind.title())
            self.operation_started(kind, self._execution_id, stage)
            self._operation_automatic = True
            self._completed_automatic_source = completed_source
        if kind != self._operation_kind:
            raise RuntimeError(f'Kumo emitted {kind!r} progress while '
                               f'{self._operation_kind!r} was active')
        if (self._operation_automatic
                and progress is self._completed_automatic_source):
            return
        if isinstance(progress, AutoTrainerProgress):
            progress_snapshot = copy.deepcopy(progress)
            self._training = progress_snapshot
            try:
                current, best = _metric_snapshots(progress_snapshot)
            except Exception:
                logger.warning('Could not derive Kumo training metrics',
                               exc_info=True)
                current, best = {}, {}
            self._current_metrics = current
            for name, value in best.items():
                lowered = name.lower()
                minimize = any(token in lowered
                               for token in ('loss', 'error', 'mae', 'mse',
                                             'rmse'))
                previous = self._best_metrics.get(name)
                if previous is None:
                    self._best_metrics[name] = value
                elif minimize:
                    self._best_metrics[name] = min(previous, value)
                else:
                    self._best_metrics[name] = max(previous, value)
        elif isinstance(progress, PredictionProgress):
            self._prediction = copy.deepcopy(progress)
        if self._operation_automatic:
            # Partitioned prediction may construct multiple loggers.
            # Only explicit operation boundaries establish ownership.
            self._active_automatic_source = progress
        self._publish()
        if self._operation_automatic and self._progress_complete(kind):
            self.operation_finished(kind, self._execution_id, None)

    def _progress_complete(self, kind: str) -> bool:
        if kind == 'training' and self._training is not None:
            return (self._training.total_trials > 0
                    and self._training.completed_trials
                    >= self._training.total_trials)
        if kind == 'prediction' and self._prediction is not None:
            return (self._prediction.total_iterations > 0
                    and self._prediction.completed_iterations
                    >= self._prediction.total_iterations)
        return False

    @_synchronized
    def operation_finished(self, kind: str, operation_id: str,
                           error: BaseException | None) -> None:
        if (kind, operation_id) != (self._operation_kind, self._operation_id):
            raise RuntimeError('Kumo progress operation ownership changed')
        stage = f'{kind.title()} failed' if error else f'{kind.title()} complete'
        self.stage_changed(
            stage,
            None if error is None else _safe_exception_text(error),
        )
        if self._operation_automatic:
            self._completed_automatic_source = self._active_automatic_source
        self._active_automatic_source = None
        self._operation_active = False
        self._operation_automatic = False

    @_synchronized
    def terminal(self, succeeded: bool, detail: str | None = None, *,
                 cancelled: bool = False) -> dict[str, Any]:
        if self._operation_active:
            assert self._operation_kind is not None
            stage = (f'{self._operation_kind.title()} complete'
                     if succeeded else f'{self._operation_kind.title()} failed')
            self.stage_changed(stage, None if succeeded else detail)
            self._operation_active = False
            self._operation_automatic = False
        status = (JobStatus.CANCELLED if cancelled else
                  JobStatus.DONE if succeeded else JobStatus.FAILED)
        stage = 'Cancelled' if cancelled else 'Completed' if succeeded else 'Failed'
        self._events.append(
            JobEventLogEntry(
                stage_name=stage,
                last_updated_at=_now(),
                detail=detail,
            ))
        self._publish(force=True, status=status)
        return self._payload(status)


@dataclasses.dataclass
class _ExecutionContext:
    store: StageStore
    config: Mapping[str, Any]
    observer: _RemoteObserver
    published_model: PublishedModel | None = None


def _decode_argument(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_argument(item) for item in value]
    if isinstance(value, Mapping):
        node = _parse_transport_node(value)
        if node is not None:
            kind, payload = node
            if kind == 'published_model':
                return _published_model_from_payload(payload)
            if kind == 'mapping':
                return {
                    str(key): _decode_argument(item)
                    for key, item in payload.items()
                }
            raise ValueError(f'unsupported Kumo argument transport {kind!r}')
        return {key: _decode_argument(item) for key, item in value.items()}
    return value


def _diagnostic(error: BaseException) -> dict[str, Any]:
    code = getattr(error, 'errno', None)
    if not isinstance(code, int) or isinstance(code, bool):
        code = None

    def text_attribute(name: str) -> str | None:
        value = getattr(error, name, None)
        if value is None:
            return None
        try:
            return str(value)
        except Exception:
            return None

    return {
        'type': type(error).__name__,
        'message': _safe_exception_text(error),
        'code': code,
        'sqlstate': text_attribute('sqlstate'),
        'sfqid': text_attribute('sfqid'),
        'traceback': _safe_traceback(error),
    }


def _publish_terminal_result(
    store: StageStore,
    location: str,
    result: Mapping[str, Any],
) -> None:
    """Idempotently publish one terminal envelope and reconcile lost acks."""
    last_error: Exception | None = None
    for attempt in range(_RESULT_PUBLISH_ATTEMPTS):
        try:
            store.put_json(location, result)
            return
        except Exception as error:
            last_error = error
            try:
                durable = store.get_json(location, missing_ok=True)
            except Exception:
                durable = None
                logger.warning(
                    'Could not reconcile terminal result publication for %s',
                    location,
                    exc_info=True,
                )
            if durable == result:
                return
            if durable is not None:
                raise RuntimeError(
                    'terminal result publication conflicts with an existing '
                    f'durable result at {location!r}') from error
            if attempt + 1 < _RESULT_PUBLISH_ATTEMPTS:
                time.sleep(_RESULT_PUBLISH_RETRY_S * (attempt + 1))
    raise RuntimeError(
        'workload completed but durable terminal result publication could '
        f'not be established at {location!r}') from last_error


def _result_observation_fields(
    context: _ExecutionContext,
    progress: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if progress is None:
        return {
            'published_model':
            (None if context.published_model is None else
             _published_model_payload(context.published_model)),
            'progress':
            None,
            'progress_kind':
            None,
            'status_report':
            None,
        }
    kind = progress.get('operation_kind')
    if kind == 'training':
        native_progress = progress.get('training')
        progress_kind: str | None = kind
    elif kind == 'prediction':
        native_progress = progress.get('prediction')
        progress_kind = kind
    else:
        native_progress = None
        progress_kind = None
    return {
        'published_model': (None if context.published_model is None else
                            _published_model_payload(context.published_model)),
        'progress':
        native_progress,
        'progress_kind':
        progress_kind,
        'status_report':
        progress['status_report'],
    }


def execute_user_function(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Snowflake-serialized entrypoint wrapping one user-owned function."""
    if config.get('schema_version') != _SCHEMA_VERSION:
        raise ValueError('unsupported Kumo Snowflake ML Job payload version')
    session = _active_job_session()
    store = StageStore(session, str(config['stage_name']))
    observer = _RemoteObserver(store, config)
    execution_context = _ExecutionContext(
        store=store,
        config=config,
        observer=observer,
    )
    token = runtime_progress.install(observer)
    observer_attached = True
    try:
        try:
            decoded_args = _decode_argument(list(args))
            decoded_kwargs = _decode_argument(dict(kwargs))
            returned = function(*decoded_args, **decoded_kwargs)
            observer.raise_for_policy_violation()
            if isinstance(returned, PublishedModel):
                execution_context.published_model = returned
            value = _json_value(returned)
            runtime_progress.reset(token)
            observer_attached = False
            terminal_progress = observer.terminal(True)
            result = {
                'schema_version': _SCHEMA_VERSION,
                'execution_id': str(config['execution_id']),
                'status': 'SUCCEEDED',
                'finished_at': _now().isoformat(),
                'value': value,
                'diagnostic': None,
                **_result_observation_fields(execution_context, terminal_progress),
            }
        except BaseException as error:
            cancelled = isinstance(error, (SystemExit, KeyboardInterrupt))
            if observer_attached:
                runtime_progress.reset(token)
                observer_attached = False
            try:
                terminal_progress = observer.terminal(
                    False, _safe_exception_text(error), cancelled=cancelled)
            except Exception:
                terminal_progress = None
                logger.exception(
                    'Could not finalize progress for failed Kumo execution %s',
                    config.get('execution_id'))
            try:
                observation_fields = _result_observation_fields(
                    execution_context, terminal_progress)
            except Exception:
                logger.exception('Could not serialize failure observations')
                observation_fields = {
                    'published_model': None,
                    'progress': None,
                    'progress_kind': None,
                    'status_report': None,
                }
            failure = {
                'schema_version': _SCHEMA_VERSION,
                'execution_id': str(config['execution_id']),
                'status': 'CANCELLED' if cancelled else 'FAILED',
                'finished_at': _now().isoformat(),
                'value': None,
                'diagnostic': _diagnostic(error),
                **observation_fields,
            }
            try:
                _publish_terminal_result(store, str(config['result_location']),
                                         failure)
            except Exception:
                logger.exception(
                    'Could not publish the durable failure for Kumo execution %s',
                    config.get('execution_id'))
            raise
        _publish_terminal_result(
            store,
            str(config['result_location']),
            result,
        )
        return result
    finally:
        if observer_attached:
            runtime_progress.reset(token)
