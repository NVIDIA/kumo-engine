# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""MLflow tracking for `Trainer.fit` (``pip install 'kumo-engine[mlflow]'``).

Each fit is one MLflow run: nested under the caller's active run if there is
one, else a new run in the current or named experiment. Each AutoML trial
attempt is a child run with its searched hyperparameters, per-epoch metrics
and summary metrics; the fit run gets the resolved model plan, the query and
the final metrics. Trial runs still open when the fit ends (the fit failed
mid-trial) are ended FAILED. MLflow autologging is suspended for the fit, so
it adds no runs of its own. Only global rank 0 logs. Any MLflow error is
logged once as a warning and stops tracking for the rest of that fit: only the
fit run is then ended, and open trial runs are left as they are. The fit
itself is unaffected. Concurrent fits in one process are not supported.
"""

import hashlib
import json
import logging
import os
import posixpath
import re
import time
from contextlib import ExitStack
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

from pytorch_lightning.loggers import MLFlowLogger

from kumo.engine._events import Event
from kumo.engine._tracking import FitContext
from kumo.engine._version import __version__
from kumo.engine.connector._extras import import_provider

logger = logging.getLogger(__name__)

_MAX_PARAM_KEY = 250
_MAX_PARAM_VALUE = 6000
# MLflow's portable name charset (its Windows rule, which omits ':').
_PARAM_KEY = re.compile(r'[/\w.\- ]+')


class MlflowTracking:
    r"""Records `Trainer.fit` in MLflow; pass as ``fit(tracking=...)``.

    Args:
        experiment_name: experiment for the fit run, created if missing.
            `None` uses the current experiment.
        run_name: name of the fit run. `None` lets MLflow pick one.
        log_system_metrics: sample CPU, memory, disk, network and (with
            ``nvidia-ml-py``) GPU usage on the fit run.
        tags: extra tags for the fit run.

    Raises:
        ImportError: if mlflow is not installed.
    """
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_system_metrics: bool = True,
        tags: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._mlflow = import_provider('mlflow', 'mlflow')
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_system_metrics = log_system_metrics
        self.tags = dict(tags or {})
        self._reset()

    def _reset(self) -> None:
        self._enabled = False
        self._client: Any = None
        self._parent: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._children: Dict[Tuple[int, int], str] = {}
        self._seconds: Optional[float] = None
        self._guard = ExitStack()

    def start(self, context: FitContext) -> None:
        self._reset()
        if _rank() != 0:
            return
        self._enabled = True
        try:
            autologging = import_provider('mlflow.utils.autologging_utils',
                                          'mlflow')
            # Not reentrant: leaving it clears the flag, so never nest it.
            if not getattr(autologging, '_AUTOLOGGING_GLOBALLY_DISABLED',
                           False):
                self._guard.enter_context(autologging.disable_autologging())
        except Exception:
            logger.warning(
                'Could not suspend MLflow autologging; it may log the '
                'internal trial fits as extra runs', exc_info=True)
        self._safe(self._start_parent, context)

    def trial_logger(self, trial: int, attempt: int) -> Optional[Any]:
        return self._safe(self._trial_logger, trial, attempt)

    def on_event(self, event: Event) -> None:
        if event.trial is not None:
            attempt = int(event.params.get('attempt', 1))
            if event.name == 'trial_started':
                self._safe(self._log_params,
                           self._safe(self._child, event.trial, attempt),
                           event.params)
            elif event.name == 'trial_ended':
                self._safe(self._end_trial, event.trial, attempt, event)
        elif event.name == 'fit_ended':
            self._seconds = event.metrics.get('seconds')

    def finish(self, result: Any, error: Optional[BaseException]) -> None:
        try:
            if result is not None:
                metrics = dict(result.metrics)
                if self._seconds is not None:
                    metrics['seconds'] = self._seconds
                self._safe(self._log_metrics, self._parent, metrics)
            status = 'FINISHED' if error is None else 'FAILED'
            if self._enabled:
                for run_id in self._children.values():
                    _quietly(self._client.set_terminated, run_id, 'FAILED')
            if self._parent is not None:
                active = self._mlflow.active_run()
                if active is not None and active.info.run_id == self._parent:
                    _quietly(self._mlflow.end_run, status)
                else:
                    _quietly(self._client.set_terminated, self._parent, status)
        finally:
            self._guard.close()
            self._reset()

    def _start_parent(self, context: FitContext) -> None:
        mlflow = self._mlflow
        self._client = mlflow.MlflowClient()
        experiment_id = None
        if self.experiment_name is not None:
            experiment = self._client.get_experiment_by_name(
                self.experiment_name)
            experiment_id = (experiment.experiment_id
                             if experiment else self._client.create_experiment(
                                 self.experiment_name))
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.run_name,
            nested=mlflow.active_run() is not None,
            tags={
                'kumo.engine_version': __version__,
                'kumo.run_id': context.run_id,
                'kumo.snapshot_id': context.snapshot_id,
                'kumo.training_table_id': context.training_table_id,
                **self.tags,
            },
            log_system_metrics=self.log_system_metrics,
        )
        self._parent = run.info.run_id
        self._experiment_id = run.info.experiment_id
        self._log_params(
            self._parent, {
                'task_type': context.task_type,
                'snapshot_id': context.snapshot_id,
                'training_table_id': context.training_table_id,
                'query_sha256': hashlib.sha256(
                    context.query.encode()).hexdigest(),
                **dict(_flatten(context.model_plan)),
            })
        self._client.log_text(self._parent, context.query, 'query.pql')
        self._client.log_dict(self._parent, dict(context.model_plan),
                              'model_plan.json')

    def _child(self, trial: int, attempt: int) -> str:
        key = (trial, attempt)
        if key not in self._children:
            run = self._client.create_run(
                self._experiment_id,
                run_name=f'trial-{trial}-attempt-{attempt}',
                tags={'mlflow.parentRunId': self._parent},
            )
            self._children[key] = run.info.run_id
        return self._children[key]

    def _trial_logger(self, trial: int, attempt: int) -> '_KumoMLFlowLogger':
        trial_logger = _KumoMLFlowLogger(
            self, run_id=self._child(trial, attempt),
            tracking_uri=self._mlflow.get_tracking_uri(), log_model=False)
        # Lightning resolves these lazily, outside our error handling.
        trial_logger.experiment
        trial_logger.version
        return trial_logger

    def _end_trial(self, trial: int, attempt: int, event: Event) -> None:
        run_id = self._child(trial, attempt)
        self._log_metrics(
            run_id, {
                name if name == 'seconds' else f'best_{name}': value
                for name, value in event.metrics.items()
            })
        succeeded = event.params.get('status') == 'succeeded'
        self._client.set_terminated(run_id,
                                    'FINISHED' if succeeded else 'FAILED')
        del self._children[(trial, attempt)]

    def _log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        entities = import_provider('mlflow.entities', 'mlflow')
        self._client.log_batch(
            run_id, params=[
                entities.Param(key, value)
                for key, value in ((key, _param_value(value))
                                   for key, value in params.items())
                if _valid_key(key) and len(value) <= _MAX_PARAM_VALUE
            ])

    def _log_metrics(self, run_id: Optional[str],
                     metrics: Mapping[str, float]) -> None:
        if run_id is None or not metrics:
            return
        entities = import_provider('mlflow.entities', 'mlflow')
        timestamp = int(time.time() * 1000)
        self._client.log_batch(
            run_id, metrics=[
                entities.Metric(key, float(value), timestamp, 0)
                for key, value in metrics.items()
            ])

    def _safe(self, function: Callable[..., Any], *args: Any,
              **kwargs: Any) -> Any:
        if not self._enabled:
            return None
        try:
            return function(*args, **kwargs)
        except Exception:
            logger.warning(
                'MLflow tracking failed; it is off for the rest of this fit, '
                'which continues', exc_info=True)
            self._enabled = False
            return None


class _KumoMLFlowLogger(MLFlowLogger):
    r"""Logs one trial's epoch metrics to an existing child run. The tracker,
    not Lightning, ends the run."""
    def __init__(self, tracking: MlflowTracking, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tracking = tracking

    def log_metrics(self, metrics: Mapping[str, float],
                    step: Optional[int] = None) -> None:
        self._tracking._safe(super().log_metrics, metrics, step)

    def log_hyperparams(self, params: Any) -> None:
        self._tracking._safe(super().log_hyperparams, params)

    def finalize(self, status: str = 'success') -> None:
        pass


def _rank() -> int:
    for name in ('RANK', 'LOCAL_RANK'):
        value = os.environ.get(name, '')
        if value.strip() not in ('', '0'):
            return 1
    return 0


def _flatten(value: Any, prefix: str = '') -> Iterator[Tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _flatten(item, f'{prefix}{key}.')
    else:
        yield prefix[:-1], value


def _valid_key(key: str) -> bool:
    return (len(key) <= _MAX_PARAM_KEY and _PARAM_KEY.fullmatch(key) is not None
            and posixpath.normpath(key) == key and not key.startswith('/'))


def _param_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _quietly(function: Callable[..., Any], *args: Any) -> None:
    try:
        function(*args)
    except Exception:
        logger.debug('MLflow cleanup call failed', exc_info=True)
