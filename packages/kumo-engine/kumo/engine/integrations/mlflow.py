# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""MLflow tracking for `Trainer.fit` (``pip install 'kumo-engine[mlflow]'``).
"""

import hashlib
import json
import logging
import os
import posixpath
import re
import time
import warnings
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Set, Tuple

from kumo.engine._events import Event
from kumo.engine._tracking import DROPPED_PARAMS, RunRecord
from kumo.engine.connector._extras import import_provider

logger = logging.getLogger(__name__)

_MAX_PARAM_KEY = 250
_MAX_PARAM_VALUE = 6000
# MLflow's portable name charset (its Windows rule, which omits ':').
_PARAM_KEY = re.compile(r'[/\w.\- ]+')
# MLflow's defaults (7 retries, 120 s timeout) can hold a call for minutes.
_HTTP_LIMITS = {
    'MLFLOW_HTTP_REQUEST_MAX_RETRIES': '2',
    'MLFLOW_HTTP_REQUEST_TIMEOUT': '30',
}
_PATH_CONNECTORS = ('file', 's3', 'gcs', 'adls')


class MlflowTracking:
    r"""Records `Trainer.fit` in MLflow: one run per fit (nested under the
    active run, if any) with a child run per trial attempt. Source tables
    are logged as run inputs: Databricks tables as Delta tables (Unity
    Catalog then links them to a model registered from the run), file and
    object-store sources by URI. Params MLflow rejects are counted in
    ``kumo.dropped_params``. Concurrent fits in one process are not supported.

    Args:
        experiment_name: experiment for the fit run, created if missing.
            `None` uses the current experiment.
        run_name: name of the fit run. `None` lets MLflow pick one.
        log_system_metrics: sample CPU, memory, disk, network and (with
            ``nvidia-ml-py``) GPU usage on the fit run.
        tags: extra tags for the fit run.
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
        self._outer: Optional[str] = None
        self._reset()

    def _reset(self) -> None:
        self._client: Any = None
        self._parent: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._children: Dict[Tuple[int, int], str] = {}
        self._artifacts: Set[str] = set()
        self._guard = ExitStack()

    def prepare(self) -> None:
        active = self._mlflow.active_run()
        self._outer = active.info.run_id if active is not None else None

    def start(self, record: RunRecord) -> None:
        self._reset()
        self._guard.enter_context(_limited_http())
        try:
            autologging = import_provider('mlflow.utils.autologging_utils',
                                          'mlflow')
            # Not reentrant: leaving it clears the flag.
            if not getattr(autologging, '_AUTOLOGGING_GLOBALLY_DISABLED',
                           False):
                self._guard.enter_context(autologging.disable_autologging())
        except Exception:
            logger.warning('Could not suspend MLflow autologging',
                           exc_info=True)
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
            parent_run_id=self._outer,
            tags=record.tags,
            log_system_metrics=self.log_system_metrics,
        )
        self._parent = run.info.run_id
        self._experiment_id = run.info.experiment_id
        self._log_params(self._parent, record.params)
        self._log_artifacts(record.artifacts)
        try:
            self._log_inputs(record.inputs)
        except Exception:
            logger.warning('Could not log the fit\'s source tables to MLflow',
                           exc_info=True)

    def on_trial(self, record: RunRecord, event: Event) -> None:
        key = (event.trial or 0, int(event.params['attempt']))
        if event.name == 'trial_started':
            run = self._client.create_run(
                self._experiment_id, run_name=record.name,
                tags={**record.tags, 'mlflow.parentRunId': self._parent})
            self._children[key] = run.info.run_id
            self._log_params(run.info.run_id, record.params)
        elif event.name == 'epoch_ended':
            self._log_metrics(self._children.get(key), event.metrics,
                              event.epoch or 0)
        elif event.name == 'trial_ended':
            run_id = self._children.pop(key)
            self._log_metrics(run_id, record.metrics)
            self._log_tags(run_id, record.tags)
            succeeded = event.params.get('status') == 'succeeded'
            self._client.set_terminated(run_id,
                                        'FINISHED' if succeeded else 'FAILED')

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        status = 'FAILED'
        try:
            if record is not None and self._parent is not None:
                self._log_metrics(self._parent, record.metrics)
                self._log_tags(self._parent, record.tags)
                self._log_artifacts(record.artifacts)
                if error is None:
                    status = 'FINISHED'
        finally:
            _quietly(self._mlflow.flush_async_logging)
            for run_id in self._children.values():
                _quietly(self._client.set_terminated, run_id, 'FAILED')
            if self._parent is not None:
                active = self._mlflow.active_run()
                if active is not None and active.info.run_id == self._parent:
                    _quietly(self._mlflow.end_run, status)
                else:
                    _quietly(self._client.set_terminated, self._parent, status)
            self._guard.close()
            self._reset()

    def _log_artifacts(self, artifacts: Mapping[str, str]) -> None:
        for name, text in artifacts.items():
            if name not in self._artifacts:
                self._client.log_text(self._parent, text, name)
                self._artifacts.add(name)

    def _log_inputs(self,
                    inputs: Mapping[str, Mapping[str, Optional[str]]]) -> None:
        meta = import_provider('mlflow.data.meta_dataset', 'mlflow')
        entities = self._mlflow.entities
        datasets = []
        for name, entry in inputs.items():
            source = _dataset_source(entry)
            if source is None:
                continue
            digest = hashlib.sha256(
                json.dumps(dict(entry), sort_keys=True).encode()).hexdigest()
            dataset = meta.MetaDataset(source, name=name, digest=digest[:8])
            datasets.append(
                entities.DatasetInput(
                    dataset._to_mlflow_entity(),
                    tags=[entities.InputTag('mlflow.data.context', 'train')]))
        if datasets:
            self._client.log_inputs(self._parent, datasets=datasets)

    def _log_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        entities = self._mlflow.entities
        self._client.log_batch(
            run_id, tags=[
                entities.RunTag(key, value)
                for key, value in tags.items()
            ])

    def _log_params(self, run_id: str, params: Mapping[str, str]) -> None:
        entities = self._mlflow.entities
        kept = _mlflow_params(params)
        dropped = len(params) - len(kept)
        tags = [entities.RunTag(DROPPED_PARAMS, str(dropped))] if dropped else []
        self._client.log_batch(
            run_id, params=[
                entities.Param(key, value)
                for key, value in kept.items()
            ], tags=tags)

    def _log_metrics(self, run_id: Optional[str],
                     metrics: Mapping[str, float], step: int = 0) -> None:
        if run_id is None or not metrics:
            return
        entities = self._mlflow.entities
        timestamp = int(time.time() * 1000)
        self._client.log_batch(
            run_id, metrics=[
                entities.Metric(key, float(value), timestamp, step)
                for key, value in metrics.items()
            ])


def _dataset_source(entry: Mapping[str, Optional[str]]) -> Any:
    connector, location = entry.get('connector'), entry.get('location')
    if not location or location.startswith('sql:'):
        return None
    if connector == 'databricks':
        delta = import_provider('mlflow.data.delta_dataset_source', 'mlflow')
        return delta.DeltaDatasetSource(
            delta_table_name=_unquoted(location),
            delta_table_version=_delta_version(entry))
    if connector in _PATH_CONNECTORS:
        registry = import_provider('mlflow.data.dataset_source_registry',
                                   'mlflow')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return registry.resolve_dataset_source(location)
    return None


def _unquoted(name: str) -> str:
    parts = re.findall(r'`((?:[^`]|``)*)`', name)
    return '.'.join(part.replace('``', '`') for part in parts) or name


def _delta_version(entry: Mapping[str, Optional[str]]) -> Optional[int]:
    version = entry.get('version') or ''
    head = version.split('_', 1)[0]
    if entry.get('version_kind') == 'exact' and head.isdigit():
        return int(head)
    return None


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


@contextmanager
def _limited_http() -> Iterator[None]:
    unset = [name for name in _HTTP_LIMITS if name not in os.environ]
    os.environ.update({name: _HTTP_LIMITS[name] for name in unset})
    try:
        yield
    finally:
        for name in unset:
            os.environ.pop(name, None)
