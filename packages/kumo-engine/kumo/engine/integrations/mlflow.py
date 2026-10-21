# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""MLflow tracking for `Trainer.fit` (``pip install 'kumo-engine[mlflow]'``).

Translates the engine's run records (`kumo.engine.tracking.run_record`,
`kumo.engine.tracking.trial_record`) into MLflow runs under their names. Each fit is
one MLflow run: nested under the caller's active run if there is one, else a
new run in the current or named experiment. Each AutoML trial attempt is a
child run with its searched hyperparameters, per-epoch metrics (step = epoch)
and summary metrics; Lightning's own metrics on it are prefixed
``lightning-`` (step = Lightning's global step). The fit run gets the resolved
model plan, the query, user and ``kumo.*`` tags, and at the end the final
metrics and the finish-time tags (published model ids as ``kumo.model_*``, or
``kumo.error`` when the fit failed). A trial run gets ``kumo.status`` and, on
failure, ``kumo.error`` when it ends.
Trial runs still open when the fit ends (the fit failed mid-trial) are ended
FAILED. MLflow autologging is suspended for the fit, so it adds no runs of its
own.

MLflow limits: params whose key MLflow rejects (characters outside
``[/\w.\- ]``, over 250 characters, or not a normalized relative path) or whose
value is over 6000 characters are not logged as params; the full plan is
always in the ``model_plan.json`` artifact, and the run's
``kumo.dropped_params`` tag counts the skipped ones.

Any MLflow error is logged once as a warning and stops tracking for the rest of
that fit: only the fit run is then ended, and open trial runs are left as they
are. The fit itself is unaffected. Concurrent fits in one process are not
supported.
"""

import logging
import posixpath
import re
import time
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from pytorch_lightning.loggers import MLFlowLogger

from kumo.engine._events import Event
from kumo.engine._tracking import (
    DROPPED_PARAMS,
    LIGHTNING_PREFIX,
    Capabilities,
    FitContext,
    RunRecord,
    run_record,
    trial_record,
)
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
        tags: extra tags for the fit run, under their given names.

    Raises:
        ImportError: if mlflow is not installed.
    """

    capabilities = Capabilities(nested_runs=True, system_metrics=True,
                                step_metrics=True, artifacts=True)

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
        self._context: Optional[FitContext] = None
        self._client: Any = None
        self._parent: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._children: Dict[Tuple[int, int], str] = {}
        self._trials: Dict[Tuple[int, int], List[Event]] = {}
        self._fit_ended: Optional[Event] = None
        self._dropped: Dict[str, int] = {}
        self._guard = ExitStack()

    def start(self, context: FitContext) -> None:
        self._reset()
        self._enabled = True
        self._context = context
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
        if event.trial is not None and self._context is not None:
            key = (event.trial, int(event.params['attempt']))
            events = self._trials.setdefault(key, [])
            events.append(event)
            if event.name == 'trial_started':
                self._safe(self._start_trial, key, events)
            elif event.name == 'epoch_ended' and event.epoch is not None:
                self._safe(self._log_steps, key, events, event.epoch)
            elif event.name == 'trial_ended':
                self._safe(self._end_trial, key, events, event)
        elif event.name == 'fit_ended':
            self._fit_ended = event

    def finish(self, result: Any, error: Optional[BaseException]) -> None:
        try:
            self._safe(self._log_result, result)
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
        record = run_record(context, tags=self.tags)
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
            tags=record.tags,
            log_system_metrics=self.log_system_metrics,
        )
        self._parent = run.info.run_id
        self._experiment_id = run.info.experiment_id
        self._log_params(self._parent, record.params)
        for name, text in record.artifacts.items():
            self._client.log_text(self._parent, text, name)

    def _start_trial(self, key: Tuple[int, int], events: List[Event]) -> None:
        record = trial_record(self._context, events)
        self._log_params(self._child(key, record), record.params)

    def _child(self, key: Tuple[int, int], record: RunRecord) -> str:
        if key not in self._children:
            run = self._client.create_run(
                self._experiment_id,
                run_name=record.name,
                tags={
                    **record.tags, 'mlflow.parentRunId': self._parent
                },
            )
            self._children[key] = run.info.run_id
        return self._children[key]

    def _trial_logger(self, trial: int,
                      attempt: int) -> Optional['_KumoMLFlowLogger']:
        run_id = self._children.get((trial, attempt))
        if run_id is None:
            return None
        trial_logger = _KumoMLFlowLogger(
            self, run_id=run_id, tracking_uri=self._mlflow.get_tracking_uri(),
            log_model=False, prefix=LIGHTNING_PREFIX)
        # Lightning resolves these lazily, outside our error handling.
        trial_logger.experiment
        trial_logger.version
        return trial_logger

    def _log_steps(self, key: Tuple[int, int], events: List[Event],
                   epoch: int) -> None:
        record = trial_record(self._context, events)
        self._log_metrics(
            self._children.get(key), {
                name: history[-1][1]
                for name, history in record.steps.items()
                if history[-1][0] == epoch
            }, step=epoch)

    def _end_trial(self, key: Tuple[int, int], events: List[Event],
                   event: Event) -> None:
        record = trial_record(self._context, events)
        run_id = self._children.pop(key)
        del self._trials[key]
        self._log_metrics(run_id, record.metrics)
        self._log_tags(run_id, record.tags)
        succeeded = event.params.get('status') == 'succeeded'
        self._client.set_terminated(run_id,
                                    'FINISHED' if succeeded else 'FAILED')

    def _log_result(self, result: Any) -> None:
        record = run_record(self._context, result, self._fit_ended,
                            tags=self.tags)
        self._log_metrics(self._parent, record.metrics)
        self._log_tags(self._parent, record.tags)

    def _log_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        entities = import_provider('mlflow.entities', 'mlflow')
        self._client.log_batch(
            run_id, tags=[
                entities.RunTag(key, value)
                for key, value in tags.items()
            ])

    def _log_params(self, run_id: str, params: Mapping[str, str]) -> None:
        entities = import_provider('mlflow.entities', 'mlflow')
        kept = _mlflow_params(params)
        dropped = len(params) - len(kept)
        tags = []
        if dropped:
            self._dropped[run_id] = self._dropped.get(run_id, 0) + dropped
            tags = [entities.RunTag(DROPPED_PARAMS, str(self._dropped[run_id]))]
        self._client.log_batch(
            run_id, params=[
                entities.Param(key, value)
                for key, value in kept.items()
            ], tags=tags)

    def _log_metrics(self, run_id: Optional[str],
                     metrics: Mapping[str, float], step: int = 0) -> None:
        if run_id is None or not metrics:
            return
        entities = import_provider('mlflow.entities', 'mlflow')
        timestamp = int(time.time() * 1000)
        self._client.log_batch(
            run_id, metrics=[
                entities.Metric(key, float(value), timestamp, step)
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
    r"""Logs one trial's Lightning metrics to an existing child run. The
    tracker, not Lightning, ends the run."""
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


def _mlflow_params(params: Mapping[str, str]) -> Dict[str, str]:
    return {
        key: value
        for key, value in params.items()
        if len(key) <= _MAX_PARAM_KEY and _PARAM_KEY.fullmatch(key) is not None
        and posixpath.normpath(key) == key and not key.startswith('/')
        and len(value) <= _MAX_PARAM_VALUE
    }


def _quietly(function: Callable[..., Any], *args: Any) -> None:
    try:
        function(*args)
    except Exception:
        logger.debug('MLflow cleanup call failed', exc_info=True)
