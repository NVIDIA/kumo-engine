# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""MLflow tracking and models for `Trainer.fit`
(``pip install 'kumo-engine[mlflow]'``)."""

import hashlib
import json
import logging
import os
import posixpath
import re
import time
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from kumoapi.plugin import ModelRegistryLocation, PublishedModel

from kumo.engine._events import Event
from kumo.engine._tracking import DROPPED_PARAMS, RunRecord
from kumo.engine._version import __version__
from kumo.engine.connector._extras import import_provider

if TYPE_CHECKING:
    from kumo_core.embedding_models import EmbeddingModelStore

    from kumo.engine.graph import GraphSnapshot
    from kumo.engine.trainer.trainer import Trainer, TrainingResult

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
# Model upload only, and only where the limits above apply.
_UPLOAD_HTTP = {'MLFLOW_HTTP_REQUEST_TIMEOUT': '600'}
_PATH_CONNECTORS = ('LOCAL_FILE', 'S3', 'GCS', 'ADLS')


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
        log_model: log the fitted model on the fit run (see `log_model`)
            before the fit returns. ``kumo.MLOps(registry=
            ModelRegistryLocation(...))`` also registers it, which implies
            this.
        serving_snapshot: exported snapshot URI the served model loads at
            startup (see `log_model`).

    ``run_id`` is the MLflow id of the latest fit run, once it has started.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_system_metrics: bool = True,
        tags: Optional[Mapping[str, str]] = None,
        log_model: bool = False,
        serving_snapshot: Optional[str] = None,
    ) -> None:
        self._mlflow = import_provider('mlflow', 'mlflow')
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_system_metrics = log_system_metrics
        self.tags = dict(tags or {})
        self.serving_snapshot = serving_snapshot
        self.logs_model = log_model
        self._outer: Optional[str] = None
        self.run_id: Optional[str] = None
        self._reset()

    def _reset(self) -> None:
        self._client: Any = None
        self._parent: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._children: Dict[Tuple[int, int], str] = {}
        self._artifacts: Set[str] = set()
        self._model_id: Optional[str] = None
        self._http_owned: FrozenSet[str] = frozenset()
        self._guard = ExitStack()

    def prepare(self) -> None:
        active = self._mlflow.active_run()
        self._outer = active.info.run_id if active is not None else None

    def start(self, record: RunRecord) -> None:
        self._reset()
        self.run_id = None
        self._http_owned = frozenset(name for name in _HTTP_LIMITS
                                     if name not in os.environ)
        self._guard.enter_context(_limited_http(self._http_owned))
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
            run_name=record.name or self.run_name,
            parent_run_id=self._outer,
            tags=record.tags,
            log_system_metrics=self.log_system_metrics,
        )
        self._parent = self.run_id = run.info.run_id
        self._experiment_id = run.info.experiment_id
        self._log_params(self._parent, record.params)
        self._log_artifacts(record.artifacts)
        _log_inputs(mlflow, self._client, self._parent, record.inputs)

    def log_monitoring(self, record: RunRecord) -> None:
        self.start(record)
        experiment_id, run_id = self._experiment_id, self.run_id
        self.finish(record, None)
        client = self._mlflow.MlflowClient()
        for run in client.search_runs(
                [experiment_id], filter_string=(
                    "tags.`kumo.kind` = 'prediction' AND tags.`kumo.run_id` "
                    f"= '{record.tags['kumo.prediction_run']}'"),
                max_results=1):
            client.set_tag(run.info.run_id, 'kumo.monitored', run_id)

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

    def log_model(
        self,
        result: 'TrainingResult',
        snapshot: 'GraphSnapshot',
        registry: Optional[ModelRegistryLocation] = None,
    ) -> Optional[PublishedModel]:
        run_id = self._parent
        if not (self.logs_model or registry) or run_id is None:
            return None
        upload = {
            name: value
            for name, value in _UPLOAD_HTTP.items() if name in self._http_owned
        }
        os.environ.update(upload)
        try:
            info = _log_model(self._mlflow, result, snapshot, name='model',
                              run_id=run_id,
                              serving_snapshot=self.serving_snapshot)
        finally:
            if self._parent == run_id:
                os.environ.update({name: _HTTP_LIMITS[name] for name in upload})
        if self._parent != run_id:
            logger.warning('The fit run %s ended before its model was '
                           'logged; the model is not registered', run_id)
            return None
        self._model_id = info.model_id
        return (None if registry is None else _register(
            self._mlflow, info, registry))

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        status = 'FAILED'
        try:
            if record is not None and self._parent is not None:
                self._log_metrics(self._parent, record.metrics,
                                  model_id=self._model_id)
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
                     metrics: Mapping[str, float], step: int = 0,
                     model_id: Optional[str] = None) -> None:
        if run_id is None or not metrics:
            return
        entities = self._mlflow.entities
        timestamp = int(time.time() * 1000)
        self._client.log_batch(
            run_id, metrics=[
                entities.Metric(key, float(value), timestamp, step,
                                model_id=model_id)
                for key, value in metrics.items()
            ])


def log_model(
    trainer: 'Trainer',
    *,
    snapshot: 'GraphSnapshot',
    name: str = 'model',
    run_id: Optional[str] = None,
    serving_snapshot: Optional[str] = None,
) -> Any:
    r"""Log a trainer's bundle as an MLflow pyfunc model in ``run_id`` (or
    the active run, else a new one) with its source tables as run inputs;
    returns its `ModelInfo`. Scores with
    ``predict(entities, params={'snapshot': uri})``; with
    ``serving_snapshot`` (an exported snapshot URI), each serving replica
    downloads that snapshot once at startup and scores against it when no
    ``snapshot`` param is given."""
    from kumo.engine._tracking import fit_inputs

    mlflow = import_provider('mlflow', 'mlflow')
    result = trainer.result
    if result is None:
        raise ValueError('log_model requires a fitted or loaded Trainer')
    with _in_run(mlflow, run_id) as active:
        _log_inputs(mlflow, mlflow.MlflowClient(), active,
                    fit_inputs(result.manifest))
        info = _log_model(mlflow, result, snapshot, name=name, run_id=active,
                          serving_snapshot=serving_snapshot)
    if result.metrics:
        mlflow.log_metrics(dict(result.metrics), run_id=info.run_id,
                           model_id=info.model_id)
    return info


def load_model(
    model_uri: str,
    embedding_models: Optional['EmbeddingModelStore'] = None,
) -> 'Trainer':
    r"""Restore the `Trainer` of a model `log_model` logged."""
    from kumo.engine._scratch import new_run_id, trainer_scratch_dir
    from kumo.engine.trainer.trainer import Trainer, warn_if_other_bundle

    mlflow = import_provider('mlflow', 'mlflow')
    destination = trainer_scratch_dir(new_run_id()) / 'mlflow-model'
    destination.mkdir(parents=True)
    local = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri,
                                            dst_path=str(destination)))
    model = mlflow.models.Model.load(str(local))
    bundle = model.flavors['python_function']['artifacts']['bundle']['path']
    trainer = Trainer.load(local / bundle, embedding_models)
    warn_if_other_bundle(model.metadata, trainer)
    return trainer


def _log_model(mlflow: Any, result: 'TrainingResult',
               snapshot: Optional['GraphSnapshot'], *, name: str, run_id: str,
               serving_snapshot: Optional[str]) -> Any:
    from kumo_core.artifacts import PROVENANCE_FILE

    from kumo.engine.integrations import _mlflow_model, _mlflow_model_code
    from kumo.engine.trainer._spec import read_training_spec
    from kumo.engine.trainer.trainer import registry_pointers

    if snapshot is None:
        raise ValueError('log_model needs the snapshot the model trained on')
    spec = read_training_spec(result.bundle_dir)
    signature, example = _mlflow_model.signature(snapshot,
                                                 spec.predictive_query)
    provenance = result.bundle_dir / PROVENANCE_FILE
    model_logger = logging.getLogger('mlflow.models.model')
    model_logger.addFilter(_no_serving_check)
    try:
        with _in_run(mlflow, run_id):
            return mlflow.pyfunc.log_model(
                name=name,
                python_model=_mlflow_model_code.__file__,
                model_config={
                    'columns': signature.outputs.input_names(),
                    'snapshot': serving_snapshot,
                },
                artifacts={
                    'bundle': str(result.bundle_dir),
                    **({'provenance': str(provenance)}
                       if provenance.is_file() else {}),
                },
                signature=signature,
                input_example=example,
                metadata=registry_pointers(result),
                pip_requirements=[f'kumo-engine[s3,gcs]=={__version__}'],
            )
    finally:
        model_logger.removeFilter(_no_serving_check)


def _log_inputs(mlflow: Any, client: Any, run_id: str,
                inputs: Mapping[str, Mapping[str, Optional[str]]]) -> None:
    try:
        meta = import_provider('mlflow.data.meta_dataset', 'mlflow')
        entities = mlflow.entities
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
            client.log_inputs(run_id, datasets=datasets)
    except Exception:
        logger.warning('Could not log the fit\'s source tables to MLflow',
                       exc_info=True)


def _register(mlflow: Any, info: Any,
              location: ModelRegistryLocation) -> PublishedModel:
    name = location.model_name
    version = str(mlflow.register_model(info.model_uri, name).version)
    client = mlflow.MlflowClient()
    if location.comment:
        client.update_model_version(name, version,
                                    description=location.comment)
    if location.alias:
        client.set_registered_model_alias(name, location.alias, version)
    return PublishedModel(name=name, version=version,
                          uri=f'models:/{name}/{version}')


def _no_serving_check(record: logging.LogRecord) -> bool:
    # MLflow's input-example check predicts without a snapshot; no switch.
    return not record.getMessage().startswith(
        'Failed to validate serving input example')


@contextmanager
def _in_run(mlflow: Any, run_id: Optional[str]) -> Iterator[str]:
    active = mlflow.active_run()
    if active is not None and run_id in (None, active.info.run_id):
        yield active.info.run_id
        return
    with mlflow.start_run(run_id=run_id, nested=active is not None) as run:
        yield run.info.run_id


def _dataset_source(entry: Mapping[str, Optional[str]]) -> Any:
    connector = (entry.get('connector') or '').upper()
    location = entry.get('location')
    if not location or location.startswith('sql:'):
        return None
    if connector == 'DATABRICKS':
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
def _limited_http(names: FrozenSet[str]) -> Iterator[None]:
    os.environ.update({name: _HTTP_LIMITS[name] for name in names})
    try:
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
