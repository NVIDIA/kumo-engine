# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""MLflow tracking and models for `Trainer.fit`, `Trainer.predict` and
monitoring (``pip install 'kumo-engine[mlflow]'``)."""

import hashlib
import json
import logging
import os
import posixpath
import re
import tempfile
import threading
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
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import quote

from kumo_core.instrumentation import format_error
from kumo_core.summaries import (
    Confusion,
    Curve,
    Dataset,
    Histogram,
    Scalar,
    Series,
    Summary,
    Table,
    Tag,
)
from kumoapi.plugin import BUNDLE_ID_KEY, ModelRegistryLocation, PublishedModel

from kumo.engine._events import Event
from kumo.engine._mlops import full_error
from kumo.engine._tracking import (
    DROPPED_PARAMS,
    RegistrationError,
    RunRecord,
    by_step,
    record_failure,
)
from kumo.engine._version import __version__
from kumo.engine.connector._extras import import_provider
from kumo.engine.integrations._mlflow_target import (
    SAGEMAKER_ARN,
    SageMakerMLflow,
    global_tracking_uri,
)
from kumo.engine.integrations._mlflow_target import resolve as resolve_target

__all__ = ['MlflowTracking', 'SageMakerMLflow', 'load_model', 'log_model']

if TYPE_CHECKING:
    from kumo_core.embedding_models import EmbeddingModelStore

    from kumo.engine._manifest import ModelManifest
    from kumo.engine.graph import GraphSnapshot
    from kumo.engine.trainer.trainer import Trainer

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
_PATH_CONNECTORS = ('LOCAL_FILE', 'S3', 'GCS')
_SAGEMAKER_MODEL = re.compile(r'[a-zA-Z0-9](-*[a-zA-Z0-9]){0,56}')
_UC_FORBIDDEN = re.compile(r'[ /\x00-\x1f\x7f]')


class MlflowTracking:
    r"""Records `Trainer.fit` in MLflow: one run per fit (under the active
    run, if any, on the same server) with a child run per trial attempt.
    Source tables are logged as run inputs: Databricks tables as Delta
    tables (Unity Catalog then links them to a model registered from the
    run), file and object-store sources by URI. Params MLflow rejects are
    counted in ``kumo.dropped_params``. Concurrent fits in one process are
    not supported. As ``kumo.MLOps(tracking=...)``, it also records each
    prediction run and each monitoring call as one run named after it,
    placed like a fit run; ``run_name`` and ``tags`` apply to fit runs only.

    Args:
        experiment_name: experiment for the fit run, created if missing.
            `None` uses MLflow's current experiment; required to record
            runs at a ``target``.
        run_name: name of the fit run. `None` names it after its registry
            model and UTC start, as ``fit-users-churn-20261027-0618``.
        log_system_metrics: sample CPU, memory, disk, network and (with
            ``nvidia-ml-py``) GPU usage on the fit run.
        tags: extra tags for the fit run.
        log_model: log the fitted model on the fit run (see `log_model`)
            before the fit returns. ``kumo.MLOps(registry=
            ModelRegistryLocation(...))`` also registers it, which implies
            this.
        target: where to record, with the credentials that reach it: a
            ``DatabricksConnector`` (Databricks MLflow and Unity Catalog,
            as its identity), a `SageMakerMLflow`, or another MLflow
            server's tracking URI. `None` uses MLflow's own tracking and
            registry URIs, read once here.

    With explicit credentials (a Databricks service principal, an AWS
    access key or role), each MLflow call sets the state MLflow reads them
    from for its duration (Databricks: ``DATABRICKS_HOST`` and a short-lived
    ``DATABRICKS_TOKEN``; SageMaker: the process-wide tracking URI), so do
    not call MLflow or Databricks yourself on other threads during a fit.
    Ambient credentials (a notebook, job, profile or instance role) set no
    credentials, but a Databricks or SageMaker target still sets the
    process-wide tracking URI for each call: Unity Catalog downloads and the
    SageMaker plugin route with it.

    ``run_id`` is the MLflow id of the most recent tracked run, once it has
    started, and ``run_url`` its link in the MLflow UI, where the server has
    one.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_system_metrics: bool = True,
        tags: Optional[Mapping[str, str]] = None,
        log_model: bool = False,
        *,
        target: Any = None,
    ) -> None:
        self._mlflow = import_provider('mlflow', 'mlflow')
        self.target = resolve_target(target)
        self._named = target is not None
        self._default_experiment: Optional[str] = (
            None if experiment_name is not None or self._named else
            import_provider('mlflow.tracking.fluent',
                            'mlflow')._get_experiment_id())
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_system_metrics = log_system_metrics
        self.tags = dict(tags or {})
        self.logs_model = log_model
        self._outer: Optional[str] = None
        self.run_id: Optional[str] = None
        self._samples: List[Tuple[Mapping[str, float], int, int]] = []
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

    @property
    def platform(self) -> str:
        return self.target.platform

    @property
    def run_url(self) -> Optional[str]:
        if self.run_id is None or self._experiment_id is None:
            return None
        return self.target.run_url(self._experiment_id, self.run_id)

    @contextmanager
    def client(self) -> Iterator[Any]:
        r"""An ``MlflowClient`` for this tracker's server, with its
        credentials and tracking URI set for the block, so your own MLflow
        calls (the client's, or ``mlflow.search_runs``) reach the server
        kumo-engine records to, as the same identity."""
        with self.target.scope(), global_tracking_uri(
                self.target.tracking_uri):
            yield self.target.client()

    def check_model_name(self, name: str) -> None:
        r"""Raises :class:`ValueError` unless this tracker's registry
        accepts ``name``."""
        check_model_name(name, self.target.registry_uri or '')

    def check(self, registry: Optional[ModelRegistryLocation] = None) -> None:
        r"""Raises before any work unless the server is reachable with these
        credentials and, given ``registry``, can register its model: one
        experiment lookup, one registry listing and, on Unity Catalog, the
        model's schema."""
        self._require_experiment()
        if registry is not None:
            self.check_model_name(registry.model_name)
        uri = self.target.tracking_uri
        owned = frozenset(name for name in _HTTP_LIMITS
                          if name not in os.environ)
        missing = False
        try:
            with _limited_http(owned), self.target.scope():
                client = self.target.client()
                if self.experiment_name is not None:
                    missing = client.get_experiment_by_name(
                        self.experiment_name) is None
                else:
                    client.search_experiments(max_results=1)
                if registry is not None:
                    client.search_registered_models(max_results=1)
                    check_registry_schema(registry.model_name,
                                          self.target.registry_uri or '')
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(f'MLflow at {uri} is not usable with these '
                               f'credentials: {full_error(error)}') from error
        if missing:
            self._create_experiment(client, owned)

    def _create_experiment(self, client: Any, owned: FrozenSet[str]) -> None:
        with _limited_http(owned), self.target.scope():
            try:
                client.create_experiment(self.experiment_name)
            except Exception as error:
                if client.get_experiment_by_name(self.experiment_name):
                    return
                raise RuntimeError(
                    f'Could not create the MLflow experiment '
                    f'{self.experiment_name!r}: {full_error(error)}. Its '
                    f'parent folder must exist: on Databricks, create it '
                    f'with `databricks workspace mkdirs <folder>` or in the '
                    f'workspace UI.') from error

    def prepare(self) -> None:
        active = self._mlflow.active_run()
        same = self._mlflow.get_tracking_uri() == self.target.tracking_uri
        self._outer = (active.info.run_id
                       if active is not None and same else None)

    def _require_experiment(self) -> None:
        if self._named and self.experiment_name is None:
            raise ValueError('MlflowTracking(target=...) records runs in an '
                             'experiment: pass experiment_name.')

    def load_model(self, location: ModelRegistryLocation,
                   **kwargs: Any) -> 'Trainer':
        r"""The `Trainer` of the version of ``location`` its alias or
        version names, from this tracker's registry."""
        if (location.alias is None) == (location.version is None):
            raise ValueError('Loading from an MLflow registry takes a '
                             'ModelRegistryLocation with an alias or a '
                             'version.')
        name, alias = location.model_name, location.alias
        reference = f'@{alias}' if alias else f'/{location.version}'
        trainer = load_model(f'models:/{name}{reference}', target=self.target,
                             **kwargs)
        version = location.version
        if alias:
            with self.target.scope():
                version = self.target.client().get_model_version_by_alias(
                    name, alias).version
        trainer._from_registry(
            PublishedModel(name, str(version), f'models:/{name}/{version}'),
            alias)
        return trainer

    def register(self, result: Any,
                 location: ModelRegistryLocation) -> PublishedModel:
        r"""Registers a fitted or loaded model (a `TrainingResult`) at
        ``location``: the version that already holds its bundle, else a new
        version of the model logged in a new run of this experiment."""
        from kumo.engine._manifest import stub
        from kumo.engine.graph import GraphSnapshot
        from kumo.engine.trainer._spec import read_training_spec
        from kumo.engine.trainer.trainer import registry_pointers

        self._require_experiment()
        client = self.target.client()
        pointers = registry_pointers(result.id)
        with _limited_http(frozenset(name for name in _HTTP_LIMITS
                                     if name not in os.environ)):
            with self.target.scope():
                held = _registered_version(client, location.model_name,
                                           pointers)
            info = None
            if held is None:
                spec = read_training_spec(result.bundle_dir)
                snapshot = GraphSnapshot(spec.snapshot_id, Path(),
                                         spec.graph_config or {})
                with self.target.scope():
                    experiment_id = self._default_experiment or '0'
                    if self.experiment_name is not None:
                        experiment = client.get_experiment_by_name(
                            self.experiment_name)
                        experiment_id = (experiment.experiment_id
                                         if experiment else
                                         client.create_experiment(
                                             self.experiment_name))
                    run_id = client.create_run(
                        experiment_id,
                        tags={'kumo.bundle_id': str(result.id)}).info.run_id
                info = _log_model(self.target, client, result.bundle_dir,
                                  result.manifest or stub('model', result.id),
                                  snapshot, name='model', run_id=run_id,
                                  experiment_id=experiment_id,
                                  serving_snapshot=None)
                with self.target.scope():
                    client.set_terminated(run_id, 'FINISHED')
            return _register(self.target, client, info, location, pointers)

    def start(self, record: RunRecord) -> None:
        self._require_experiment()
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
        except Exception as error:
            logger.warning('Could not suspend MLflow autologging: %s',
                           format_error(error))
        self._client = client = self.target.client()
        context = import_provider('mlflow.tracking.context.registry',
                                  'mlflow')
        with self.target.scope():
            experiment_id = self._default_experiment
            if self.experiment_name is not None:
                experiment = client.get_experiment_by_name(
                    self.experiment_name)
                experiment_id = (experiment.experiment_id if experiment else
                                 client.create_experiment(
                                     self.experiment_name))
            tags = dict(record.tags)
            if self._outer is not None:
                tags['mlflow.parentRunId'] = self._outer
            if record.description is not None:
                tags['mlflow.note.content'] = record.description
            run = client.create_run(
                experiment_id or '0',
                run_name=(self.run_name if self.run_name
                          and record.tags.get('kumo.kind') == 'fit' else
                          record.name),
                tags=context.resolve_tags(tags))
        self._parent = self.run_id = run.info.run_id
        self._experiment_id = run.info.experiment_id
        if self.log_system_metrics and record.tags.get('kumo.kind') == 'fit':
            self._guard.enter_context(
                _SystemMetrics(self._log_system_metrics))
        self._log_params(self._parent, record.params)
        self._log_artifacts(record.artifacts)
        self._log_samples()

    @contextmanager
    def sample_system_metrics(self) -> Iterator[None]:
        r"""Samples system metrics during the block, for the next prediction
        or monitoring run this tracker logs."""
        self._samples = []
        if not self.log_system_metrics:
            yield
            return
        with _SystemMetrics(lambda metrics, step: self._samples.append(
                (metrics, step, int(time.time() * 1000)))):
            yield

    def log_prediction_run(self, record: RunRecord) -> None:
        self.start(record)
        self.finish(record, record_failure(record))

    def log_monitoring(self, record: RunRecord) -> None:
        self.start(record)
        experiment_id, run_id = self._experiment_id, self.run_id
        client = self._client
        self.finish(record, None)
        with self.target.scope():
            for run in client.search_runs(
                    [experiment_id], filter_string=(
                        "tags.`kumo.kind` = 'prediction' AND "
                        "tags.`kumo.prediction_id` = "
                        f"'{record.tags['kumo.monitors']}'"),
                    max_results=1):
                client.set_tag(run.info.run_id, 'kumo.monitored', run_id)

    def on_trial(self, record: RunRecord, event: Event) -> None:
        key = (event.trial or 0, int(event.params['attempt']))
        if event.name == 'trial_started':
            with self.target.scope():
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
            with self.target.scope():
                self._client.set_terminated(
                    run_id, 'FINISHED' if succeeded else 'FAILED')

    def log_model(
        self,
        bundle_dir: Path,
        manifest: Optional['ModelManifest'],
        snapshot: 'GraphSnapshot',
        registry: Optional[ModelRegistryLocation] = None,
    ) -> Optional[PublishedModel]:
        from kumo.engine.trainer.trainer import registry_pointers

        run_id = self._parent
        if not (self.logs_model or registry) or run_id is None:
            return None
        upload = {
            name: value
            for name, value in _UPLOAD_HTTP.items() if name in self._http_owned
        }
        os.environ.update(upload)
        try:
            info = _log_model(self.target, self._client, bundle_dir, manifest,
                              snapshot, name='model', run_id=run_id,
                              experiment_id=self._experiment_id,
                              serving_snapshot=None)
        finally:
            if self._parent == run_id:
                os.environ.update({name: _HTTP_LIMITS[name] for name in upload})
        if self._parent != run_id:
            logger.warning('The fit run %s ended before its model was '
                           'logged; the model is not registered', run_id)
            return None
        self._model_id = info.model_id
        if registry is None:
            return None
        try:
            return _register(self.target, self._client, info, registry,
                             registry_pointers(_id(manifest)))
        except Exception as error:
            raise RegistrationError(full_error(error)) from error

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        status = 'FAILED'
        fit_run = (record is not None and self._parent is not None
                   and record.tags.get('kumo.kind') == 'fit')
        try:
            if record is not None and self._parent is not None:
                for epoch, metrics in by_step(record.steps):
                    self._log_metrics(self._parent, metrics, epoch)
                self._log_metrics(self._parent, record.metrics,
                                  model_id=self._model_id)
                self._log_tags(self._parent, record.tags)
                self._log_artifacts(record.artifacts)
                self.log_summaries(record.summaries)
                if error is None:
                    status = 'FINISHED'
        finally:
            self._guard.close()
            with self.target.scope():
                for run_id in self._children.values():
                    _quietly(self._client.set_terminated, run_id, 'FAILED')
                if self._parent is not None:
                    _quietly(self._client.set_terminated, self._parent,
                             status)
            self._reset()
        hint = _sagemaker_ui_hint(self.target.tracking_uri)
        if fit_run and hint:
            warnings.warn(hint, UserWarning, stacklevel=2)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self._log_metrics(self._parent, metrics, step)

    def _log_system_metrics(self, metrics: Mapping[str, float],
                            step: int) -> None:
        self._log_metrics(self._parent, metrics, step)

    def _log_samples(self) -> None:
        samples, self._samples = self._samples, []
        entities = self._mlflow.entities
        metrics = [
            entities.Metric(metric_key(name), float(value), timestamp, step)
            for values, step, timestamp in samples
            for name, value in values.items()
        ]
        with self.target.scope():
            for start in range(0, len(metrics), _BATCH_METRICS):
                self._client.log_batch(
                    self._parent,
                    metrics=metrics[start:start + _BATCH_METRICS])

    def log_summaries(self, summaries: Sequence[Summary]) -> None:
        r"""Logs each summary on the current run through its type's call in
        `_SINKS`; a test scalar also goes on the logged model."""
        for summary in summaries:
            _SINKS[type(summary)](self, summary)

    def _log_scalar(self, summary: Scalar) -> None:
        model_id = self._model_id if summary.name.startswith('test/') else None
        self._log_metrics(self._parent, {summary.name: summary.value},
                          model_id=model_id)

    def _log_tag(self, summary: Tag) -> None:
        self._log_tags(self._parent, {summary.name: summary.value})

    def _log_series(self, summary: Series) -> None:
        self._log_points({summary.name: summary.points})

    def _log_curve(self, summary: Curve) -> None:
        self._log_points({summary.x.name: summary.x.points,
                          summary.y.name: summary.y.points})

    def _log_histogram(self, summary: Histogram) -> None:
        self._log_points({
            f'{summary.name}/count': tuple(enumerate(summary.counts)),
            f'{summary.name}/lower_edge': tuple(enumerate(summary.edges, 1)),
        })

    def _log_table(self, summary: Table) -> None:
        path = f'{summary.name}.json'
        self._log_artifacts({path: json.dumps({
            'columns': list(summary.columns),
            'data': [list(row) for row in summary.rows],
        }, default=str)})
        with self.target.scope():
            tags = self._client.get_run(self._parent).data.tags
            logged = json.loads(tags.get('mlflow.loggedArtifacts', '[]'))
            entry = {'path': path, 'type': 'table'}
            if entry not in logged:
                self._client.set_tag(self._parent, 'mlflow.loggedArtifacts',
                                     json.dumps([*logged, entry]))

    def _log_confusion(self, summary: Confusion) -> None:
        self._log_table(Table(
            summary.name, ('actual', *summary.labels),
            tuple((label, *row)
                  for label, row in zip(summary.labels, summary.counts))))

    def _log_dataset(self, summary: Dataset) -> None:
        source = _dataset_source(summary.source)
        if source is None:
            return
        entities = self._mlflow.entities
        version = summary.source.get('version') or ''
        dataset = entities.Dataset(
            name=summary.name,
            digest=version[:36] or hashlib.sha256(json.dumps(
                dict(summary.source), sort_keys=True).encode()).hexdigest()[:8],
            source_type=source._get_source_type(),
            source=source.to_json(),
            schema=json.dumps({'mlflow_colspec': [
                {'name': name, 'type': _COLUMN_TYPES.get(column['dtype'],
                                                         'string')}
                for name, column in summary.schema.items()]}),
            profile=json.dumps(summary.profile, default=str))
        with self.target.scope():
            self._client.log_inputs(self._parent, datasets=[
                entities.DatasetInput(dataset, tags=[entities.InputTag(
                    'mlflow.data.context', summary.context)])])

    def _log_points(self, series: Mapping[str, Sequence[Tuple[int, float]]]
                    ) -> None:
        entities = self._mlflow.entities
        timestamp = int(time.time() * 1000)
        metrics = [
            entities.Metric(metric_key(name), float(value), timestamp, step)
            for name, points in series.items() for step, value in points
        ]
        with self.target.scope():
            for start in range(0, len(metrics), _BATCH_METRICS):
                self._client.log_batch(
                    self._parent,
                    metrics=metrics[start:start + _BATCH_METRICS])

    def _log_artifacts(self, artifacts: Mapping[str, str]) -> None:
        for name, text in artifacts.items():
            if name in self._artifacts:
                continue
            with self.target.scope():
                repository = self.target.artifact_repository(
                    self._client.get_run(self._parent).info.artifact_uri
                ) if self.target.explicit else None
                if repository is None:
                    self._client.log_text(self._parent, text, name)
                else:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / posixpath.basename(name)
                        path.write_text(text)
                        repository.log_artifact(
                            str(path), posixpath.dirname(name) or None)
            self._artifacts.add(name)

    def _log_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        entities = self._mlflow.entities
        with self.target.scope():
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
        with self.target.scope():
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
        with self.target.scope():
            self._client.log_batch(
                run_id, metrics=[
                    entities.Metric(metric_key(key), float(value), timestamp,
                                    step, model_id=model_id)
                    for key, value in metrics.items()
                ])


_SINKS: Dict[type, Callable[[MlflowTracking, Any], None]] = {
    Scalar: MlflowTracking._log_scalar,
    Tag: MlflowTracking._log_tag,
    Series: MlflowTracking._log_series,
    Curve: MlflowTracking._log_curve,
    Histogram: MlflowTracking._log_histogram,
    Table: MlflowTracking._log_table,
    Confusion: MlflowTracking._log_confusion,
    Dataset: MlflowTracking._log_dataset,
}
_BATCH_METRICS = 1000  # MLflow's per-request limit
_SYSTEM_METRICS_FAILURES = 6
_COLUMN_TYPES = {'int64': 'long', 'int32': 'integer', 'float64': 'double',
                 'float32': 'float', 'bool': 'boolean', 'date': 'datetime',
                 'time': 'datetime'}


class _SystemMetrics:
    r"""Samples MLflow's CPU, disk, network and GPU monitors every
    ``interval`` seconds on its own thread and hands each aggregate to
    ``log(metrics, step)``, which logs it through the tracker's
    credentials. (MLflow's own ``SystemMetricsMonitor`` logs with
    process-wide ones.)"""

    def __init__(self, log: Callable[[Mapping[str, float], int], None],
                 interval: float = 10.0) -> None:
        self._log = log
        self._interval = interval
        self._monitors = _metric_monitors()
        self._failed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='kumo-system-metrics')

    def __enter__(self) -> '_SystemMetrics':
        if self._monitors:
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(self._interval + 5)

    def _run(self) -> None:
        step = failures = 0
        while not self._stop.wait(self._interval):
            metrics: Dict[str, float] = {}
            for monitor in self._monitors:
                monitor.collect_metrics()
                metrics.update(monitor.aggregate_metrics())
                monitor.clear_metrics()
            try:
                self._log({f'system/{k}': v
                           for k, v in metrics.items()}, step)
                failures = 0
            except Exception as error:
                if not self._failed:
                    logger.info('Could not log system metrics, skipping that '
                                'reading: %s', format_error(error))
                self._failed = True
                failures += 1
                if failures == _SYSTEM_METRICS_FAILURES:
                    logger.info('System metrics stopped after %d failed logs '
                                'in a row', failures)
                    return
            step += 1


def _metric_monitors() -> List[Any]:
    monitors = []
    for module, name in (('cpu_monitor', 'CPUMonitor'),
                         ('disk_monitor', 'DiskMonitor'),
                         ('network_monitor', 'NetworkMonitor'),
                         ('gpu_monitor', 'GPUMonitor')):
        try:
            monitors.append(
                getattr(
                    import_provider(f'mlflow.system_metrics.metrics.{module}',
                                    'mlflow'), name)())
        except Exception:
            logger.debug('No %s system metrics', name, exc_info=True)
    return monitors


def check_model_name(name: str, registry_uri: str) -> None:
    r"""Raises :class:`ValueError` unless the MLflow registry at
    ``registry_uri`` accepts ``name``: SageMaker managed MLflow takes letters,
    digits and hyphens; Unity Catalog takes ``catalog.schema.model``, each
    part at most 255 characters without spaces, ``/``, ``.`` or control
    characters."""
    is_databricks_unity_catalog_uri = import_provider(
        'mlflow.utils.uri', 'mlflow').is_databricks_unity_catalog_uri

    if SAGEMAKER_ARN.fullmatch(registry_uri or ''):
        if not _SAGEMAKER_MODEL.fullmatch(name):
            raise ValueError(
                f'SageMaker MLflow cannot register a model named {name!r}: '
                f'use letters, digits and hyphens, starting and ending with '
                f'a letter or digit, at most 57 letters and digits '
                f'(^[a-zA-Z0-9](-*[a-zA-Z0-9]){{0,56}}$), e.g. '
                f"'users-churn'.")
    elif is_databricks_unity_catalog_uri(registry_uri or ''):
        parts = name.split('.')
        if len(parts) > 3 or not all(parts) or any(
                len(part) > 255 or _UC_FORBIDDEN.search(part)
                for part in parts):
            raise ValueError(
                f'Unity Catalog cannot register a model named {name!r}: use '
                f'catalog.schema.model (on a cluster, a shorter name takes '
                f'its current catalog and schema), each part at most 255 '
                f'characters with no space, "/" or control character, e.g. '
                f"'main.ml.users_churn'.")


def _sagemaker_ui_hint(tracking_uri: str) -> Optional[str]:
    match = SAGEMAKER_ARN.fullmatch(tracking_uri or '')
    if match is None:
        return None
    region, kind, name = match.groups()
    command = (f'aws sagemaker create-presigned-mlflow-tracking-server-url '
               f'--tracking-server-name {name} --region {region}'
               if kind == 'mlflow-tracking-server' else
               f'aws sagemaker create-presigned-mlflow-app-url '
               f'--arn {tracking_uri} --region {region}')
    return ("MLflow's run links do not open on SageMaker: open the MLflow UI "
            f'from SageMaker Studio or at the URL `{command}` prints.')


def metric_key(name: str) -> str:
    r"""The MLflow name of a Kumo metric: MLflow rejects ``@``, so
    ``map@10`` is logged as ``map_at_10``."""
    return name.replace('@', '_at_')


def log_model(
    trainer: 'Trainer',
    *,
    snapshot: 'GraphSnapshot',
    name: str = 'model',
    run_id: Optional[str] = None,
    serving_snapshot: Any = None,
    target: Any = None,
    experiment_name: Optional[str] = None,
) -> Any:
    r"""Log a trainer's bundle as an MLflow pyfunc model in ``run_id`` (or
    the active run, else a new one in ``experiment_name``) with its source
    tables as run inputs; returns its `ModelInfo`. ``target`` says where,
    with what credentials, as in `MlflowTracking`; `None` uses MLflow's own
    tracking URI. Scores with ``predict(entities, params={'snapshot':
    uri})``; with ``serving_snapshot`` (an exported snapshot URI, or a
    `GraphSnapshot` or snapshot id stored under the artifact root of the
    enclosing `kumo.mlops` block), each serving replica downloads that
    snapshot once at startup and scores against it when no ``snapshot``
    param is given."""
    from kumo.engine._manifest import stub
    from kumo.engine._mlops import snapshot_uri
    from kumo.engine._tracking import fit_inputs

    mlflow = import_provider('mlflow', 'mlflow')
    result = trainer.result
    if result is None:
        raise ValueError('log_model requires a fitted or loaded Trainer')
    tracking = MlflowTracking(experiment_name, target=target)
    target = tracking.target
    if run_id is None and tracking._named and experiment_name is None:
        tracking._require_experiment()
    client = target.client()
    active = mlflow.active_run()
    if run_id is None and active is not None and (
            mlflow.get_tracking_uri() == target.tracking_uri):
        run_id = active.info.run_id
    created = run_id is None
    with target.scope():
        if created:
            experiment = (client.get_experiment_by_name(
                tracking.experiment_name) if tracking.experiment_name else
                          None)
            run_id = client.create_run(
                experiment.experiment_id if experiment else
                tracking._default_experiment or '0').info.run_id
        experiment_id = client.get_run(run_id).info.experiment_id
        _log_inputs(mlflow, client, run_id, fit_inputs(result.manifest))
    info = _log_model(target, client, result.bundle_dir, result.manifest or
                      (None if result.id is None else stub('model', result.id)),
                      snapshot, name=name, run_id=run_id,
                      experiment_id=experiment_id,
                      serving_snapshot=snapshot_uri(serving_snapshot))
    with target.scope():
        if result.metrics:
            client.log_batch(run_id, metrics=[
                mlflow.entities.Metric(metric_key(key), float(value),
                                       int(time.time() * 1000), 0,
                                       model_id=info.model_id)
                for key, value in result.metrics.items()
            ])
        if created:
            client.set_terminated(run_id, 'FINISHED')
    return info


def load_model(
    model_uri: str,
    embedding_models: Optional['EmbeddingModelStore'] = None,
    *,
    target: Any = None,
    accelerator: Literal['cpu', 'gpu', 'mps', 'auto'] = 'auto',
) -> 'Trainer':
    r"""Restore the `Trainer` of a model `log_model` logged, from the
    server ``target`` reaches, as in `MlflowTracking` (`None`: MLflow's own
    tracking URI)."""
    from kumo.engine._scratch import new_run_id, trainer_scratch_dir
    from kumo.engine.trainer.trainer import Trainer, warn_if_other_bundle

    mlflow = import_provider('mlflow', 'mlflow')
    target = resolve_target(target)
    destination = trainer_scratch_dir(new_run_id()) / 'mlflow-model'
    destination.mkdir(parents=True)
    with target.scope():
        local = Path(_download(mlflow, target, model_uri, destination))
    model = mlflow.models.Model.load(str(local))
    bundle = model.flavors['python_function']['artifacts']['bundle']['path']
    trainer = Trainer.load(local / bundle, embedding_models,
                           accelerator=accelerator)
    warn_if_other_bundle(model.metadata, trainer)
    return trainer


def _download(mlflow: Any, target: Any, model_uri: str,
              destination: Path) -> str:
    if target.explicit and model_uri.startswith('models:/'):
        client = target.client()
        source = model_uri
        reference = model_uri[len('models:/'):]
        name, at, alias = reference.partition('@')
        if at:
            source = client.get_model_version_by_alias(name, alias).source
        elif '/' in reference:
            name, _, version = reference.partition('/')
            source = client.get_model_version(name, version).source
        if source.startswith('models:/') and '/' not in source[8:]:
            location = client.get_logged_model(source[8:]).artifact_location
            repository = target.artifact_repository(location)
            if repository is not None:
                return repository.download_artifacts('', str(destination))
    return mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=str(destination),
        tracking_uri=target.tracking_uri, registry_uri=target.registry_uri)


def _log_model(target: Any, client: Any, bundle_dir: Path,
               manifest: Optional['ModelManifest'],
               snapshot: Optional['GraphSnapshot'], *, name: str, run_id: str,
               experiment_id: Optional[str],
               serving_snapshot: Optional[str]) -> Any:
    r"""What ``mlflow.pyfunc.log_model`` does, through ``client``: it only
    logs to the process-wide tracking URI. A bundle without a training spec
    is logged without a signature."""
    from kumo_core.artifacts import PROVENANCE_FILE

    from kumo.engine.integrations import _mlflow_model, _mlflow_model_code
    from kumo.engine.trainer._spec import (
        SPEC_FILE,
        listed_file,
        read_training_spec,
    )
    from kumo.engine.trainer.trainer import registry_pointers

    mlflow = import_provider('mlflow', 'mlflow')
    if snapshot is None:
        raise ValueError('log_model needs the snapshot the model trained on')
    bundle_dir = Path(bundle_dir)
    signature, example = None, None
    if (bundle_dir / SPEC_FILE).is_file():
        signature, example = _mlflow_model.signature(
            snapshot,
            read_training_spec(bundle_dir).predictive_query)
    pointers = registry_pointers(_id(manifest))
    provenance = bundle_dir / PROVENANCE_FILE
    with target.scope():
        model = client.create_logged_model(
            experiment_id or client.get_run(run_id).info.experiment_id,
            name=name, source_run_id=run_id,
            params=dict(client.get_run(run_id).data.params))
    mlflow_model = mlflow.models.Model(
        artifact_path=model.artifact_location, model_uuid=model.model_id,
        run_id=run_id, metadata=pointers, model_id=model.model_id)
    model_logger = logging.getLogger('mlflow.models.model')
    model_logger.addFilter(_no_serving_check)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = str(Path(tmp) / 'model')
            mlflow.pyfunc.save_model(
                local, mlflow_model=mlflow_model,
                python_model=_mlflow_model_code.__file__,
                model_config={
                    'columns': [] if signature is None else
                    signature.outputs.input_names(),
                    'snapshot': serving_snapshot,
                },
                artifacts={
                    'bundle': str(bundle_dir),
                    **({'provenance': str(provenance)}
                       if provenance.is_file() else {}),
                },
                signature=signature,
                input_example=example,
                metadata=pointers,
                pip_requirements=[
                    _requirement(listed_file(bundle_dir, 'encoders'))
                ],
            )
            with target.scope():
                repository = target.artifact_repository(
                    model.artifact_location)
                if repository is None:
                    client.log_model_artifacts(model.model_id, local)
                else:
                    repository.log_artifacts(local)
                logged = client.finalize_logged_model(model.model_id,
                                                      'READY')
                client.log_outputs(run_id, [
                    mlflow.entities.LoggedModelOutput(model.model_id, 0)
                ])
    except BaseException:
        with target.scope():
            _quietly(client.finalize_logged_model, model.model_id, 'FAILED')
        raise
    finally:
        model_logger.removeFilter(_no_serving_check)
    return mlflow_model.get_model_info(logged)


def _id(manifest: Optional['ModelManifest']) -> Optional[str]:
    return None if manifest is None else manifest.id


def _requirement(encoders: Optional[Path]) -> str:
    from kumo_core.encoders.config import EncodersConfig
    from kumo_core.io.json import load_json
    from kumoapi.encoder import TextEmbedding

    extras = ['s3', 'gcs']
    path = None if encoders is None else str(encoders)
    config = (load_json(path, EncodersConfig)
              if path and Path(path).is_file() else None)
    if config is not None and any(
            isinstance(encoder, TextEmbedding)
            for table in config.encoders.values()
            for encoder in table.values()):
        extras.append('text')
    return f"kumo-engine[{','.join(extras)}]=={__version__}"


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
    except Exception as error:
        logger.warning('Could not log the fit\'s source tables to MLflow: %s',
                       format_error(error), exc_info=True)


def _register(target: Any, client: Any, info: Any,
              location: ModelRegistryLocation,
              metadata: Mapping[str, str]) -> PublishedModel:
    name = location.model_name
    with target.scope():
        version = _registered_version(client, name, metadata)
        if version is None:
            try:
                client.create_registered_model(name)
            except Exception as error:
                if _code(error) != 'RESOURCE_ALREADY_EXISTS':
                    raise
            version = str(
                client.create_model_version(name, info.model_uri,
                                            run_id=info.run_id, tags=metadata,
                                            model_id=info.model_id).version)
        else:
            logger.info('%s version %s already holds bundle %s; reused it',
                        name, version, metadata.get(BUNDLE_ID_KEY))
        if location.comment:
            client.update_model_version(name, version,
                                        description=location.comment)
        if location.alias:
            client.set_registered_model_alias(name, location.alias, version)
    return PublishedModel(name=name, version=version,
                          uri=f'models:/{name}/{version}',
                          metadata=dict(metadata))


def _registered_version(client: Any, name: str,
                        metadata: Mapping[str, str]) -> Optional[str]:
    r"""The version of ``name`` that already holds this bundle, if any."""
    bundle = metadata.get(BUNDLE_ID_KEY)
    if not bundle:
        return None
    token = None
    while True:
        try:
            page = client.search_model_versions(f"name = '{name}'",
                                                page_token=token)
        except Exception as error:
            if _code(error) == 'RESOURCE_DOES_NOT_EXIST':
                return None
            raise
        for version in page:
            # Unity Catalog's search results carry tags as a method.
            tags = version.tags
            if not isinstance(tags, Mapping) or not tags:
                tags = client.get_model_version(name, version.version).tags
            if (tags or {}).get(BUNDLE_ID_KEY) == bundle:
                return str(version.version)
        token = page.token
        if not token:
            return None


def _code(error: BaseException) -> str:
    return str(getattr(error, 'error_code', '') or '')


def check_registry_schema(name: str, registry_uri: str) -> None:
    r"""On Unity Catalog, raises :class:`ValueError` unless the catalog and
    schema of a ``catalog.schema.model`` name exist and these credentials
    can use them (one metadata call, no warehouse)."""
    uri = import_provider('mlflow.utils.uri', 'mlflow')
    parts = name.split('.')
    if not uri.is_databricks_unity_catalog_uri(registry_uri) or len(
            parts) != 3:
        return
    databricks = import_provider('mlflow.utils.databricks_utils', 'mlflow')
    rest = import_provider('mlflow.utils.rest_utils', 'mlflow')
    schema = f'{parts[0]}.{parts[1]}'
    response = rest.http_request(
        databricks.get_databricks_host_creds(registry_uri),
        f'/api/2.1/unity-catalog/schemas/{quote(schema, safe="")}', 'GET',
        max_retries=1, timeout=30)
    if response.status_code in (403, 404):
        try:
            body = response.json()
        except ValueError:
            body = {}
        raise ValueError(
            f"Unity Catalog cannot register {name!r}: "
            f"{body.get('error_code', response.status_code)}: "
            f"{body.get('message', response.reason)}. The catalog and schema "
            f"must exist, and the registering principal needs USE CATALOG, "
            f"USE SCHEMA and CREATE MODEL on them.")
    response.raise_for_status()


def _no_serving_check(record: logging.LogRecord) -> bool:
    # MLflow's input-example check predicts without a snapshot; no switch.
    return not record.getMessage().startswith(
        'Failed to validate serving input example')


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
