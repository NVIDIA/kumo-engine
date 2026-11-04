# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""MLflow tracking and model logging for `Trainer.fit`."""

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
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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
    cast,
)
from urllib.parse import quote, urlparse

from kumo_core.instrumentation import format_error
from kumo_core.summaries import Dataset, Note, Summary, Table, Tag
from kumoapi.plugin import BUNDLE_ID_KEY, ModelRegistryLocation, PublishedModel

from kumo.engine import _card
from kumo.engine._events import Event
from kumo.engine._mlops import full_error
from kumo.engine._run_names import (
    is_dated_fit,
    model_label,
    record_name,
    registered_fit_name,
    trial_name,
)
from kumo.engine._summary_tables import Routed, route, table_json
from kumo.engine._text import metric_key
from kumo.engine._tracking import (
    DROPPED_PARAMS,
    RegistrationError,
    RunRecord,
    SingleRunHooks,
    TrackingBase,
    by_step,
    fit_run_id,
    model_metrics,
)
from kumo.engine._version import __version__
from kumo.engine.connector._extras import import_provider
from kumo.engine.integrations._mlflow_target import (
    SAGEMAKER_ARN,
    SageMakerMLflow,
    environ,
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
_PARAM_KEY = re.compile(r'[/\w.\- ]+')
_HTTP_LIMITS = {
    'MLFLOW_HTTP_REQUEST_MAX_RETRIES': '2',
    'MLFLOW_HTTP_REQUEST_TIMEOUT': '30',
}
_UPLOAD_HTTP = {'MLFLOW_HTTP_REQUEST_TIMEOUT': '600'}
_PATH_CONNECTORS = ('LOCAL_FILE', 'S3', 'GCS')
_SAGEMAKER_MODEL = re.compile(r'[a-zA-Z0-9](-*[a-zA-Z0-9]){0,56}')
_UC_FORBIDDEN = re.compile(r'[ /\x00-\x1f\x7f]')
_HINTED: set = set()
_BATCH_METRICS = 1000
_SYSTEM_METRICS_FAILURES = 6
_COLUMN_TYPES = {'int64': 'long', 'int32': 'integer', 'float64': 'double',
                 'float32': 'float', 'bool': 'boolean', 'date': 'datetime',
                 'time': 'datetime'}
_Reading = Tuple[Mapping[str, float], int, int]  # values, second, time
_FLUSH_SECONDS = 10.0
_MIN_READINGS = 2


class MlflowTracking(SingleRunHooks, TrackingBase):
    r"""Records `Trainer.fit` in MLflow, one run per fit.

    Args:
        target: a ``DatabricksConnector``, `SageMakerMLflow` or tracking URI.
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
        self.logged_run_name: Optional[str] = None
        self.experiment: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._samples: List[_Reading] = []
        self._sampled_run: Optional[Tuple[Any, str]] = None
        self._reset()

    def _reset(self) -> None:
        self._client: Any = None
        self._parent: Optional[str] = None
        self._children: Dict[Tuple[int, int], str] = {}
        self._trials: Dict[Tuple[int, int], str] = {}
        self._artifacts: Set[str] = set()
        self._card: List[Summary] = []
        self._model_id: Optional[str] = None
        self._published: Optional[PublishedModel] = None
        self._fit: Tuple[Any, Any] = (None, None)
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
        r"""An ``MlflowClient`` with this tracker's credentials."""
        with self.target.scope(), global_tracking_uri(
                self.target.tracking_uri):
            yield self.target.client()

    def check_model_name(self, name: str) -> None:
        r"""Raises :class:`ValueError` unless the registry accepts ``name``."""
        check_model_name(name, self.target.registry_uri or '')

    def check(self, registry: Optional[ModelRegistryLocation] = None) -> None:
        r"""Raises unless the server (and ``registry``) is usable."""
        self._require_experiment()
        if registry is not None:
            self.check_model_name(registry.model_name)
        uri = self.target.tracking_uri
        missing = False
        try:
            with _limited_http(), self.target.scope():
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
            raise _unusable(uri, error) from error
        if not missing:
            return
        with _limited_http(), self.target.scope():
            try:
                client.create_experiment(self.experiment_name)
            except Exception as error:
                if client.get_experiment_by_name(self.experiment_name):
                    return
                raise RuntimeError(
                    f'Could not create the MLflow experiment '
                    f'{self.experiment_name!r}: {full_error(error)}; create '
                    f'its parent folder first.') from error

    def prepare(self) -> None:
        active = self._mlflow.active_run()
        same = self._mlflow.get_tracking_uri() == self.target.tracking_uri
        self._outer = (active.info.run_id
                       if active is not None and same else None)

    def _experiment_for(self, client: Any,
                        name: Optional[str] = None) -> str:
        name = name or self.experiment_name
        if name is None:
            return self._default_experiment or '0'
        experiment = client.get_experiment_by_name(name)
        return (experiment.experiment_id
                if experiment else client.create_experiment(name))

    def _require_experiment(self) -> None:
        if self._named and self.experiment_name is None:
            raise ValueError('MlflowTracking(target=...) needs '
                             'experiment_name.')

    def load_model(self, location: ModelRegistryLocation,
                   **kwargs: Any) -> 'Trainer':
        r"""The `Trainer` of ``location``'s alias or version."""
        if (location.alias is None) == (location.version is None):
            raise ValueError('Pass a ModelRegistryLocation with either an '
                             'alias or a version.')
        name, alias, version = (location.model_name, location.alias,
                                location.version)
        if alias:
            with self.target.scope():
                version = self.target.client().get_model_version_by_alias(
                    name, alias).version
        uri = f'models:/{name}/{version}'
        trainer = load_model(uri, target=self.target, **kwargs)
        trainer._from_registry(PublishedModel(name, str(version), uri), alias)
        return trainer

    def register(self, result: Any,
                 location: ModelRegistryLocation) -> PublishedModel:
        r"""Registers a `TrainingResult` at ``location``."""
        from kumo.engine.graph import GraphSnapshot
        from kumo.engine.trainer._provenance import bundle_manifest
        from kumo.engine.trainer._run_spec import registry_metrics
        from kumo.engine.trainer._spec import read_training_spec
        from kumo.engine.trainer.trainer import registry_pointers

        self._require_experiment()
        self._client = client = self.target.client()
        pointers = registry_pointers(result.id)
        metrics = registry_metrics(result)
        with _limited_http():
            with self.target.scope():
                held = _registered_version(client, location.model_name,
                                           pointers)
            fit, model = self._fit_of(fit_run_id(result.manifest))
            info = None if model is None else SimpleNamespace(
                model_uri=f'models:/{model.model_id}', run_id=fit.run_id,
                model_id=model.model_id)
            if held is None and info is None:
                spec = read_training_spec(result.bundle_dir)
                snapshot = GraphSnapshot(spec.snapshot_id, Path(),
                                         spec.graph_config or {})
                info = _log_model(
                    self.target, client, result.bundle_dir,
                    bundle_manifest(result.bundle_dir, result.id), snapshot,
                    name=_logged_model_name(location.model_name,
                                            result.manifest),
                    run_id=None if fit is None else fit.run_id,
                    experiment_id=(self._registry_experiment(
                        location.model_name) if fit is None else
                                   fit.experiment_id), serving_snapshot=None)
                if fit is not None:
                    self._log_metrics(fit.run_id, metrics,
                                      model_id=info.model_id)
            published = _register(self.target, client, info, location,
                                  pointers, held)
            if fit is None:
                _describe_version(self.target, client, published,
                                  _version_text(metrics, None, None, None,
                                                True))
            else:
                if held is None:
                    self._rename_fit(fit, published)
                self._describe(published, fit.run_id, metrics)
            return published

    def _logged_model(self, fit: Any) -> Any:
        try:
            with self.target.scope():
                models = self._client.search_logged_models(
                    [fit.experiment_id],
                    filter_string=f"source_run_id = '{fit.run_id}'",
                    max_results=1)
        except Exception:
            logger.debug('No logged model of %s', fit.run_id, exc_info=True)
            return None
        return models[0] if models else None

    def _registry_experiment(self, model_name: str) -> str:
        r"""The experiment for a model logged in no run, never Default."""
        name = self.experiment_name or model_label(model_name, None, None)
        if self.experiment_name is None and self.platform == 'databricks':
            name = f'/Shared/kumo/{name}'
        with self.target.scope():
            return self._experiment_for(self._client, name)

    def _rename_fit(self, fit: Any, published: PublishedModel) -> None:
        if not is_dated_fit(fit.run_name):
            return
        name = registered_fit_name(published.name, published.version)
        with self.target.scope():
            self._client.update_run(fit.run_id, name=name)
            for trial in self._client.search_runs(
                    [fit.experiment_id], filter_string=(
                        f"tags.mlflow.parentRunId = '{fit.run_id}'")):
                self._client.update_run(
                    trial.info.run_id,
                    name=trial.info.run_name.replace(fit.run_name, name, 1))
        self._log_tags(fit.run_id, {'kumo.model_name': published.name,
                                    'kumo.model_version': published.version})
        self._update_card(fit.run_id)

    def start(self, record: RunRecord) -> None:
        self._require_experiment()
        self._reset()
        self.run_id = None
        self._http_owned = self._guard.enter_context(_limited_http())
        try:
            autologging = import_provider('mlflow.utils.autologging_utils',
                                          'mlflow')
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
            experiment = (client.get_experiment_by_name(self.experiment_name)
                          if self.experiment_name is not None else
                          client.get_experiment(self._default_experiment
                                                or '0'))
            if experiment is None:
                experiment = client.get_experiment(
                    client.create_experiment(self.experiment_name))
            experiment_id, self.experiment = (experiment.experiment_id,
                                              experiment.name)
            note = experiment.tags.get(_NOTE)
            if note != _EXPERIMENT_NOTE and (
                    note.startswith(('kumo-engine:', 'Runs from kumo-engine'))
                    if note else
                    not client.search_runs([experiment_id], max_results=1)):
                client.set_experiment_tag(experiment_id, _NOTE,
                                          _EXPERIMENT_NOTE)
            tags = dict(record.tags)
            if self._outer is not None:
                tags['mlflow.parentRunId'] = self._outer
            kind = record.tags.get('kumo.kind')
            name = (record_name(record, self.run_name)
                    if self.run_name and kind in ('fit', 'prediction')
                    else record.name)
            if kind == 'prediction' and name:
                name = _free_name(client, experiment_id, name)
            run = client.create_run(
                experiment_id, run_name=name,
                start_time=_millis(record.started),
                tags=context.resolve_tags(tags))
        self._parent = self.run_id = run.info.run_id
        self.logged_run_name = run.info.run_name
        self._experiment_id = run.info.experiment_id
        if self._samples_system() and record.tags.get('kumo.kind') == 'fit':
            self._guard.enter_context(_SystemMetrics(
                lambda readings: self._log_readings(client, run.info.run_id,
                                                    readings)))
        self._log_params(self._parent, record.params)
        self._log_artifacts(record.artifacts)
        self._sampled_run = (self._client, self._parent)
        self._log_samples(*self._sampled_run)

    @contextmanager
    def sample_system_metrics(self) -> Iterator[None]:
        r"""Samples system metrics for the next prediction run."""
        self._samples, self._sampled_run = [], None
        if not self._samples_system():
            yield
            return
        with _SystemMetrics(lambda readings: self._samples.extend(readings)):
            yield
        if self._sampled_run is not None:
            self._log_samples(*self._sampled_run)

    def _samples_system(self) -> bool:
        return self.log_system_metrics and os.environ.get(
            'MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING', '').strip().lower() not in (
                'false', '0')

    def log_monitoring(self, record: RunRecord) -> None:
        r"""Folds a monitoring call into the prediction run it checks."""
        self.run_id = None
        prediction_id = record.tags.get('kumo.prediction_id')
        with _limited_http():
            self._client = self.target.client()
            run_id = record.prediction_run
            if run_id is None:
                found = self._prediction_run(prediction_id)
                if found is None:
                    raise LookupError(f'No MLflow prediction run has '
                                      f'kumo.prediction_id {prediction_id}')
                run_id = found.info.run_id
            self._fold(run_id, self._name_previous(record))

    def _prediction_run(self, prediction_id: Optional[str]) -> Optional[Any]:
        with self.target.scope():
            experiment = (None if self.experiment_name is None else
                          self._client.get_experiment_by_name(
                              self.experiment_name))
            found = self._client.search_runs(
                [experiment.experiment_id if experiment else
                 self._default_experiment or '0'],
                filter_string=("tags.`kumo.kind` = 'prediction' AND "
                               "tags.`kumo.prediction_id` = "
                               f"'{prediction_id}'"),
                max_results=1)
        return found[0] if found else None

    def _name_previous(self, record: RunRecord) -> RunRecord:
        previous = next((summary.value for summary in record.summaries
                         if isinstance(summary, Tag)
                         and summary.name == 'kumo.drift_previous'), None)
        if previous is None or not any(
                isinstance(summary, Note) and previous in summary.text
                for summary in record.summaries):
            return record
        found = self._prediction_run(previous)
        if found is None:
            return record
        return replace(record, summaries=tuple(
            replace(summary, text=summary.text.replace(
                previous, found.info.run_name))
            if isinstance(summary, Note) else summary
            for summary in record.summaries))

    def _fold(self, run_id: str, record: RunRecord) -> None:
        routed = route(record.summaries)
        self._log_routed(routed, run_id)
        self._update_card(
            run_id, 'realized' if record.tags.get('kumo.monitor')
            == 'evaluate' else 'drift', routed.card)
        with self.target.scope():
            run = self._client.get_run(run_id).info
        self._experiment_id, self.logged_run_name = (run.experiment_id,
                                                     run.run_name)
        self.run_id = run_id

    def on_trial(self, record: RunRecord, event: Event) -> None:
        key = (event.trial or 0, int(event.params['attempt']))
        if event.name == 'trial_started':
            with self.target.scope():
                run = self._client.create_run(
                    self._experiment_id, run_name=trial_name(
                        *key, cast(str, self.logged_run_name)),
                    tags={**record.tags, 'mlflow.parentRunId': self._parent})
            self._children[key] = self._trials[key] = run.info.run_id
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
                              snapshot, name=_logged_model_name(
                                  registry and registry.model_name, manifest),
                              run_id=run_id,
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
        pointers = registry_pointers(_id(manifest))
        try:
            with self.target.scope():
                held = _registered_version(self._client,
                                           registry.model_name, pointers)
            self._published = _register(self.target, self._client, info,
                                        registry, pointers, held)
        except Exception as error:
            raise RegistrationError(full_error(error)) from error
        return self._published

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        status = 'FAILED'
        kind = None if record is None else record.tags.get('kumo.kind')
        started = self._parent is not None
        fit_run = started and kind == 'fit'
        ended = None if record is None else _millis(record.finished)
        try:
            if record is not None and self._parent is not None:
                for epoch, metrics in by_step(record.steps):
                    self._log_metrics(self._parent, metrics, epoch)
                self._log_summaries(record.summaries)
                self._log_metrics(self._parent, record.metrics,
                                  model_id=self._model_id)
                self._log_tags(self._parent, record.tags)
                self._log_artifacts(record.artifacts)
                self._log_files(record.files)
                if kind == 'prediction':
                    self._fit = self._fit_of(record.tags.get('kumo.fit_run_id'))
                    self._log_model_input()
                if fit_run and not self.run_name:
                    self._rename(record)
                if fit_run and self._published is not None:
                    self._describe_version(record)
                self._write_card(record)
                if error is None:
                    status = 'FINISHED'
        finally:
            self._guard.close()
            with self.target.scope():
                for run_id in self._children.values():
                    _quietly(self._client.set_terminated, run_id, 'FAILED')
                if self._parent is not None:
                    _quietly(self._client.set_terminated, self._parent,
                             status, ended)
            self._reset()
        hint = _sagemaker_ui_hint(self.target.tracking_uri)
        if (hint and started and kind in ('fit', 'prediction')
                and self.target.tracking_uri not in _HINTED):
            _HINTED.add(self.target.tracking_uri)
            warnings.warn(f'MLflow run {self.logged_run_name!r} in experiment '
                          f'{self.experiment!r}: {hint}', UserWarning,
                          stacklevel=2)

    def _update_card(self, run_id: str, block: Optional[str] = None,
                     summaries: Sequence[Summary] = ()) -> None:
        self._client = self._client or self.target.client()
        with self.target.scope():
            run = self._client.get_run(run_id)
        split = self._editable(run, first=False)
        if split is None:
            return
        card, user = split
        card = _card.refreshed(card, run.data.metrics, run.data.tags,
                               run.data.params)
        items = list(summaries)

        def render(kept: Sequence[Summary], files: Sequence[str]) -> str:
            return card if block is None else _with_block(
                card, block, _card.block(kept, [
                    item for item in items
                    if isinstance(item, Table) and item not in kept]))

        def minimal(files: Sequence[str]) -> str:
            return '\n'.join([_card.shortened(card, files), *_card.TAIL])

        self._set_card(run, render, minimal, items, user, _budget(user))

    def _write_card(self, record: RunRecord) -> None:
        kind = record.tags.get('kumo.kind')
        if kind not in ('fit', 'prediction'):
            return
        try:
            with self.target.scope():
                run = self._client.get_run(self._parent)
            split = self._editable(run, first=True)
            if split is None:
                return
            blocks = ([_part(name, '') for name in _BLOCKS]
                      if kind == 'prediction' else [])
            card = _card.run_card(
                run.data.tags, run.data.params, run.data.metrics,
                model_metrics(record), record.description,
                [*self._card, *_card.charts(run.data.metrics)],
                *self._fit_link(), blocks=blocks)

            def render(kept: Sequence[Summary], files: Sequence[str]) -> str:
                return '\n'.join([replace(card, summaries=kept).text(files),
                                  *_card.TAIL])

            def minimal(files: Sequence[str]) -> str:
                return '\n'.join([card.minimal(files), *_card.TAIL])

            user = split[1]
            self._set_card(run, render, minimal, card.summaries, user,
                           _budget(user)
                           - (_MONITORING_BYTES if blocks else 0))
        except Exception as error:
            logger.warning('Could not write the description of MLflow run '
                           '%s: %s', self._parent, format_error(error))

    def _editable(self, run: Any, first: bool) -> Optional[Tuple[str, str]]:
        tags = run.data.tags
        if tags.get('kumo.card_frozen') == 'true':
            return None
        description = tags.get(_NOTE, '')
        end = _CARD_END.search(description)
        if end and _card_hash(description[:end.start()]) == end.group(1):
            return description[:end.start()].rstrip('\n'), description[
                end.end():]
        if first and end is None:
            return '', description and f'\n\n{description}'
        with self.target.scope():
            self._client.set_tag(run.info.run_id, 'kumo.card_frozen', 'true')
        warnings.warn(
            f'The description of MLflow run {run.info.run_name!r} was edited '
            f"above kumo's card, so kumo no longer updates it.", UserWarning,
            stacklevel=2)
        return None

    def _set_card(self, run: Any,
                  render: Callable[[Sequence[Summary], Sequence[str]], str],
                  minimal: Callable[[Sequence[str]], str],
                  items: Sequence[Summary], user: str, budget: int) -> None:
        run_id = run.info.run_id
        files = [entry['path'] for entry in json.loads(
            run.data.tags.get('mlflow.loggedArtifacts', '[]'))]
        kept = list(items)
        while True:
            card = render(kept, files)
            tables = [item for item in kept if isinstance(item, Table)]
            if _size(card) <= budget or not tables:
                break
            largest = max(tables, key=lambda t: len(_card.table_text(t)))
            kept.remove(largest)
            self._write_tables(run_id, [replace(largest, brief=False)])
            files.append(f'{largest.name}.json')
        if _size(card) > budget:
            card = minimal(files)
        with self.target.scope():
            self._client.set_tag(run_id, _NOTE,
                                 _sealed(card) + user)

    def _rename(self, record: RunRecord) -> None:
        version = record.tags.get('kumo.model_version')
        if version is None:
            return
        name = registered_fit_name(record.tags['kumo.model_name'], version)
        self.logged_run_name = name
        with self.target.scope():
            self._client.update_run(cast(str, self._parent), name=name)
            for key, run_id in self._trials.items():
                self._client.update_run(run_id, name=trial_name(*key, name))

    def _describe_version(self, record: RunRecord) -> None:
        published = cast(PublishedModel, self._published)
        try:
            self._describe(published, cast(str, self._parent),
                           model_metrics(record))
        except Exception as error:
            logger.warning('Could not describe %s version %s: %s',
                           published.name, published.version,
                           format_error(error))

    def _describe(self, published: PublishedModel, fit_id: str,
                  metrics: Mapping[str, float]) -> None:
        with self.target.scope():
            run = self._client.get_run(fit_id)
        _describe_version(self.target, self._client, published, _version_text(
            metrics, run.data.tags.get('kumo.tune_metric'), run.info.run_name,
            run.data.tags.get('kumo.bundle_uri'), not self._unity()))

    def _fit_of(self, fit_id: Optional[str]) -> Tuple[Any, Any]:
        r"""The fit run's info and its logged model, each None if missing."""
        if not fit_id:
            return None, None
        try:
            with self.target.scope():
                fit = self._client.get_run(fit_id).info
        except Exception:
            logger.debug('No fit run %s here', fit_id, exc_info=True)
            return None, None
        return fit, self._logged_model(fit)

    def _unity(self) -> bool:
        return import_provider('mlflow.utils.uri', 'mlflow'
                               ).is_databricks_unity_catalog_uri(
                                   self.target.registry_uri or '')

    def _log_model_input(self) -> None:
        if self._fit[1] is None:
            return
        entities = self._mlflow.entities
        with self.target.scope():
            self._client.log_inputs(self._parent, models=[
                entities.LoggedModelInput(self._fit[1].model_id)])

    def _fit_link(self) -> Tuple[Optional[str], Optional[str]]:
        fit = self._fit[0]
        if fit is None:
            return None, None
        return fit.run_name, (
            self.target.run_url(fit.experiment_id, fit.run_id)
            if self.platform == 'databricks' else
            f'#/experiments/{fit.experiment_id}/runs/{fit.run_id}')

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self._log_metrics(self._parent, metrics, step)

    def _log_samples(self, client: Any, run_id: str) -> None:
        samples, self._samples = self._samples, []
        self._log_readings(client, run_id, samples)

    def _log_readings(self, client: Any, run_id: str,
                      readings: Sequence['_Reading']) -> None:
        entities = self._mlflow.entities
        self._log_batches(client, run_id, [
            entities.Metric(metric_key(name), float(value), timestamp, step)
            for values, step, timestamp in readings
            for name, value in values.items()
        ])

    def _log_batches(self, client: Any, run_id: str,
                     metrics: Sequence[Any]) -> None:
        with self.target.scope():
            for start in range(0, len(metrics), _BATCH_METRICS):
                client.log_batch(
                    run_id, metrics=metrics[start:start + _BATCH_METRICS])

    def _log_summaries(self, summaries: Sequence[Summary]) -> None:
        routed = route(summaries)
        self._card += routed.card
        self._log_routed(routed, self._parent)

    def _log_routed(self, routed: Routed, run_id: str) -> None:
        self._log_inputs(run_id, routed.datasets)
        scalars = routed.scalars
        self._log_metrics(run_id, {s.name: s.value for s in scalars
                                   if s.role == 'model'},
                          model_id=self._model_id)
        self._log_metrics(run_id, {s.name: s.value for s in scalars
                                   if s.role != 'model'})
        if routed.charts:
            timestamp = int(time.time() * 1000)
            self._log_readings(self._client, run_id, [
                (values, step, timestamp)
                for step, values in by_step(routed.charts)])
        if routed.tags:
            self._log_tags(run_id, routed.tags)
        self._write_tables(run_id, routed.tables)

    def _log_inputs(self, run_id: str, summaries: Sequence[Dataset]) -> None:
        datasets = [d for d in map(self._dataset_input, summaries) if d]
        if datasets:
            with self.target.scope():
                try:
                    self._client.log_inputs(run_id, datasets=datasets)
                except self._mlflow.exceptions.MlflowException:
                    self._client.log_inputs(run_id, datasets=datasets)

    def _dataset_input(self, summary: Dataset) -> Any:
        source = _dataset_source(summary.source)
        if source is None or summary.context == 'output':
            return None
        entities = self._mlflow.entities
        dataset = entities.Dataset(
            name=summary.name,
            digest=_digest(summary.name, summary.source, summary.schema),
            source_type=source._get_source_type(),
            source=source.to_json(),
            schema=json.dumps({'mlflow_colspec': [
                {'name': name, 'type': _COLUMN_TYPES.get(column['dtype'],
                                                         'string')}
                for name, column in summary.schema.items()]}),
            profile=json.dumps({
                key: value for key, value in summary.profile.items()
                if not isinstance(value, (Mapping, list, tuple))
            }, default=str))
        return entities.DatasetInput(
            dataset, tags=[entities.InputTag('mlflow.data.context',
                                             summary.context)])

    def _write_tables(self, run_id: str, tables: Sequence[Table]) -> None:
        if not tables:
            return
        for table in tables:
            self._write_text(run_id, f'{table.name}.json', table_json(table))
        with self.target.scope():
            tags = self._client.get_run(run_id).data.tags
            logged = json.loads(tags.get('mlflow.loggedArtifacts', '[]'))
            entries = [{'path': f'{table.name}.json', 'type': 'table'}
                       for table in tables]
            added = [entry for entry in entries if entry not in logged]
            if added:
                self._client.set_tag(run_id, 'mlflow.loggedArtifacts',
                                     json.dumps([*logged, *added]))

    def _log_artifacts(self, artifacts: Mapping[str, str]) -> None:
        for name, text in artifacts.items():
            if name not in self._artifacts:
                self._write_text(self._parent, name, text)
                self._artifacts.add(name)

    def _log_files(self, files: Mapping[str, str]) -> None:
        for name, directory in files.items():
            with self.target.scope():
                repository = self.target.artifact_repository(
                    self._client.get_run(self._parent).info.artifact_uri
                ) if self.target.explicit else None
                if repository is None:
                    self._client.log_artifacts(self._parent, directory, name)
                else:
                    repository.log_artifacts(directory, name)

    def _write_text(self, run_id: str, name: str, text: str) -> None:
        with self.target.scope():
            repository = self.target.artifact_repository(
                self._client.get_run(run_id).info.artifact_uri
            ) if self.target.explicit else None
            if repository is None:
                self._client.log_text(run_id, text, name)
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / posixpath.basename(name)
                    path.write_text(text)
                    repository.log_artifact(str(path),
                                            posixpath.dirname(name) or None)

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
        self._log_batches(self._client, run_id, [
            entities.Metric(metric_key(key), float(value), timestamp, step,
                            model_id=model_id)
            for key, value in metrics.items()
        ])


def _digest(name: str, source: Mapping[str, Optional[str]],
            schema: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        [name, dict(source), dict(schema)], sort_keys=True,
        default=str).encode()).hexdigest()[:8]


_NOTE = 'mlflow.note.content'
_EXPERIMENT_NOTE = (
    "Runs from kumo-engine. Fits: `tags.kumo.kind = 'fit'`. Daily "
    "predictions: `tags.kumo.kind = 'prediction'`, sorted by the `anchors` "
    'param.')
_CARD_END = re.compile(r'<!-- /kumo:card sha256=([0-9a-f]{16}) -->')
_BLOCKS = ('drift', 'realized')
_MONITORING_BYTES = 800


def _card_hash(card: str) -> str:
    lines = card.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    text = '\n'.join(line.rstrip() for line in lines).strip('\n')
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _sealed(card: str) -> str:
    return f'{card}\n<!-- /kumo:card sha256={_card_hash(card)} -->'


def _size(card: str) -> int:
    return len(_sealed(card).encode())


def _budget(user: str) -> int:
    return min(4500, 5000 - len(user.encode()))


def _free_name(client: Any, experiment_id: str, name: str) -> str:
    candidate, suffix = name, 1
    while True:
        quote_char = "'" if "'" not in candidate else '"'
        if quote_char in candidate or not client.search_runs(
                [experiment_id], max_results=1, filter_string=(
                    'attributes.run_name = '
                    f'{quote_char}{candidate}{quote_char}')):
            return candidate
        suffix += 1
        candidate = f'{name}-{suffix}'


def _millis(moment: Optional[datetime]) -> Optional[int]:
    return None if moment is None else int(moment.timestamp() * 1000)


def _part(name: str, text: str) -> str:
    return (f'<!-- kumo:{name} -->\n\n{text}\n\n<!-- /kumo:{name} -->' if text
            else f'<!-- kumo:{name} --><!-- /kumo:{name} -->')


def _with_block(card: str, name: str, text: str) -> str:
    part = _part(name, text)
    existing = re.compile(f'<!-- kumo:{name} -->.*?<!-- /kumo:{name} -->',
                          re.DOTALL)
    if existing.search(card):
        return existing.sub(lambda _: part, card, count=1)
    head, notes, rest = card.rpartition(_card.NOTES_LINE)
    return f'{head}{part}\n\n{notes}{rest}' if notes else f'{card}\n{part}'


class _SystemMetrics:
    def __init__(self, log: Callable[[List[_Reading]], None],
                 interval: Optional[float] = None) -> None:
        fixed = os.environ.get('MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL')
        self._log = log
        self._interval = interval or float(fixed or 1)
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
            self._thread.join(_FLUSH_SECONDS + 5)

    def _run(self) -> None:
        began = flushed = time.monotonic()
        readings: List[_Reading] = []
        taken = failures = 0
        step = -1
        stopped = False
        while not stopped:
            stopped = self._stop.wait(self._interval)
            metrics: Dict[str, float] = {}
            for monitor in self._monitors:
                monitor.collect_metrics()
                metrics.update(monitor.aggregate_metrics())
                monitor.clear_metrics()
            step = max(round(time.monotonic() - began), step + 1)
            readings.append(({f'system/{k}': v for k, v in metrics.items()},
                             step, int(time.time() * 1000)))
            taken += 1
            if taken < _MIN_READINGS or not (
                    stopped or time.monotonic() - flushed >= _FLUSH_SECONDS):
                continue
            batch, readings, flushed = readings, [], time.monotonic()
            try:
                self._log(batch)
                failures = 0
            except Exception as error:
                if not self._failed:
                    logger.info('Could not log system metrics, skipping those '
                                'readings: %s', format_error(error))
                self._failed = True
                failures += 1
                if failures == _SYSTEM_METRICS_FAILURES:
                    logger.info('System metrics stopped after %d failed logs '
                                'in a row', failures)
                    return


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
    r"""Raises :class:`ValueError` unless ``registry_uri`` accepts ``name``."""
    is_databricks_unity_catalog_uri = import_provider(
        'mlflow.utils.uri', 'mlflow').is_databricks_unity_catalog_uri

    if SAGEMAKER_ARN.fullmatch(registry_uri or ''):
        if not _SAGEMAKER_MODEL.fullmatch(name):
            raise ValueError(
                f'SageMaker MLflow cannot register a model named {name!r}: '
                f'use up to 57 letters and digits joined by hyphens, e.g. '
                f"'users-churn'.")
    elif is_databricks_unity_catalog_uri(registry_uri or ''):
        parts = name.split('.')
        if len(parts) > 3 or not all(parts) or any(
                len(part) > 255 or _UC_FORBIDDEN.search(part)
                for part in parts):
            raise ValueError(
                f'Unity Catalog cannot register a model named {name!r}: use '
                f'catalog.schema.model without spaces or "/", e.g. '
                f"'main.ml.users_churn'.")


def _unusable(uri: str, error: Exception) -> RuntimeError:
    if (urlparse(uri).scheme in ('', 'file')
            and 'MLFLOW_ALLOW_FILE_STORE' in str(error)):
        return RuntimeError(
            f'MLflow refused the local folder store at {uri}: use its default '
            f'sqlite:///mlflow.db or another tracking backend.')
    return RuntimeError(f'MLflow at {uri} is not usable: {full_error(error)}')


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


def log_model(
    trainer: 'Trainer',
    *,
    snapshot: 'GraphSnapshot',
    name: Optional[str] = None,
    run_id: Optional[str] = None,
    serving_snapshot: Any = None,
    target: Any = None,
    experiment_name: Optional[str] = None,
) -> Any:
    r"""Logs a trainer as an MLflow pyfunc model; returns its `ModelInfo`.

    Args:
        serving_snapshot: snapshot to score against when the request names
            none.
        target: as in `MlflowTracking`.
    """
    from kumo.engine._manifest import FIT_ROLES, artifact_summaries
    from kumo.engine._mlops import snapshot_uri
    from kumo.engine.trainer._provenance import bundle_manifest
    from kumo.engine.trainer._run_spec import registry_metrics

    mlflow = import_provider('mlflow', 'mlflow')
    result = trainer.result
    if result is None:
        raise ValueError('log_model requires a fitted or loaded Trainer')
    tracking = MlflowTracking(experiment_name, target=target)
    target = tracking.target
    if run_id is None:
        tracking._require_experiment()
    client = tracking._client = target.client()
    active = mlflow.active_run()
    if run_id is None and active is not None and (
            mlflow.get_tracking_uri() == target.tracking_uri):
        run_id = active.info.run_id
    created = run_id is None
    with target.scope():
        if created:
            run_id = client.create_run(
                tracking._experiment_for(client)).info.run_id
        experiment_id = client.get_run(run_id).info.experiment_id
    summaries = [] if result.manifest is None else artifact_summaries(
        result.manifest, FIT_ROLES)
    try:
        tracking._log_inputs(
            run_id, [s for s in summaries if isinstance(s, Dataset)])
    except Exception as error:
        logger.warning('Could not log the fit\'s source tables to MLflow: %s',
                       format_error(error), exc_info=True)
    registered = getattr(result.registered_model, 'name', None)
    info = _log_model(target, client, result.bundle_dir,
                      bundle_manifest(result.bundle_dir, result.id),
                      snapshot, name=name or _logged_model_name(
                          registered, result.manifest), run_id=run_id,
                      experiment_id=experiment_id,
                      serving_snapshot=snapshot_uri(serving_snapshot))
    tracking._log_metrics(run_id, registry_metrics(result),
                          model_id=info.model_id)
    if created:
        with target.scope():
            client.set_terminated(run_id, 'FINISHED')
    return info


def load_model(
    model_uri: str,
    embedding_models: Optional['EmbeddingModelStore'] = None,
    *,
    target: Any = None,
    accelerator: Literal['cpu', 'gpu', 'mps', 'auto'] = 'auto',
) -> 'Trainer':
    r"""Restores the `Trainer` of a model `log_model` logged.

    Args:
        target: as in `MlflowTracking`.
    """
    from kumo.engine._deps import assets_through
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
    with assets_through(target.assets, local / bundle, _target_hint):
        trainer = Trainer.load(local / bundle, embedding_models,
                               accelerator=accelerator)
    warn_if_other_bundle(model.metadata, trainer)
    return trainer


def _target_hint(reader: str, location: str) -> str:
    target = {
        'DatabricksConnector': 'DatabricksConnector(identity=...)',
        'S3Connector': 'SageMakerMLflow(arn, identity=S3Identity(...))',
    }.get(reader, f'a {reader}')
    return (f'pass target={target} to load_model(model_uri, ...), with an '
            f'identity that reads {location.rsplit("/", 2)[0]}')


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
    elif target.platform == 'mlflow' and model_uri.startswith('models:/'):
        model_uri = _registry_download_uri(target.client(), model_uri)
    return mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=str(destination),
        tracking_uri=target.tracking_uri, registry_uri=target.registry_uri)


def _registry_download_uri(client: Any, model_uri: str) -> str:
    r"""MLflow's own models:/ resolution, against the target's registry.

    MLflow resolves models:/ with the process-wide registry URI and ignores
    the one passed to download_artifacts.
    """
    models = import_provider('mlflow.store.artifact.utils.models', 'mlflow')
    resolved = models.get_model_name_and_version(client, model_uri)
    if len(resolved) == 1:
        return client.get_logged_model(resolved[0]).artifact_location
    return client.get_model_version_download_uri(*resolved)


def _logged_model_name(registered: Optional[str], manifest: Any) -> str:
    r"""model_label without the characters MLflow rejects."""
    table = getattr(getattr(manifest, 'parents', None), 'training_table',
                    None)
    return re.sub(r'[/:.%"\']', '-', model_label(
        registered, getattr(table, 'query', None),
        getattr(table, 'task_type', None)))


def _log_model(target: Any, client: Any, bundle_dir: Path,
               manifest: Optional['ModelManifest'],
               snapshot: Optional['GraphSnapshot'], *, name: str,
               run_id: Optional[str], experiment_id: Optional[str],
               serving_snapshot: Optional[str]) -> Any:
    from kumo_core.artifacts import PROVENANCE_FILE

    from kumo.engine.integrations import _mlflow_model, _mlflow_model_code
    from kumo.engine.trainer._spec import SPEC_FILE, read_training_spec
    from kumo.engine.trainer.trainer import _encoders, registry_pointers

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
        run = None if run_id is None else client.get_run(run_id)
        model = client.create_logged_model(
            experiment_id or run.info.experiment_id, name=name,
            source_run_id=run_id,
            params=None if run is None else dict(run.data.params))
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
                    'assets_from': None,
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
                    _requirement(_encoders(bundle_dir))
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
                if run_id is not None:
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


def _requirement(config: Any) -> str:
    from kumoapi.encoder import TextEmbedding

    extras = ['s3', 'gcs']
    if config is not None and any(
            isinstance(encoder, TextEmbedding)
            for table in config.encoders.values()
            for encoder in table.values()):
        extras.append('text')
    return f"kumo-engine[{','.join(extras)}]=={__version__}"


def _register(target: Any, client: Any, info: Any,
              location: ModelRegistryLocation,
              metadata: Mapping[str, str],
              version: Optional[str]) -> PublishedModel:
    r"""Registers ``info`` as a new version unless ``version`` holds it."""
    name = location.model_name
    with target.scope():
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


def _version_text(metrics: Mapping[str, float], tune: Optional[str],
                  fit: Optional[str], bundle: Optional[str],
                  with_metrics: bool) -> str:
    trained = ' '.join([*([f'Trained by `{fit}`.'] if fit else []),
                        *([f'Bundle `{bundle}`.'] if bundle else [])])
    return '\n'.join(filter(None, [
        _card.version_headline(metrics, tune) if with_metrics else '',
        trained]))


def _describe_version(target: Any, client: Any, published: PublishedModel,
                      text: str) -> None:
    name, version = published.name, published.version
    with target.scope():
        current = client.get_model_version(name, version).description or ''
        end = _VERSION_END.search(current)
        if end and _card_hash(current[:end.start()]) != end.group(1):
            return
        user = current[end.end():] if end else current and f'\n\n{current}'
        described = (f'{text}\n<!-- /kumo:version sha256='
                     f'{_card_hash(text)} -->{user}')
        if len(described) <= 5000 and described != current:
            client.update_model_version(name, version, description=described)


_VERSION_END = re.compile(r'<!-- /kumo:version sha256=([0-9a-f]{16}) -->')


def _registered_version(client: Any, name: str,
                        metadata: Mapping[str, str]) -> Optional[str]:
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
            # Unity Catalog returns tags as a method.
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
    r"""On Unity Catalog, raises unless the model's schema is usable."""
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
            f"{body.get('message', response.reason)}; create the schema or "
            f"grant USE CATALOG, USE SCHEMA and CREATE MODEL on it.")
    response.raise_for_status()


def _no_serving_check(record: logging.LogRecord) -> bool:
    return not record.getMessage().startswith(
        'Failed to validate serving input example')


def _dataset_source(entry: Mapping[str, Optional[str]]) -> Any:
    connector = (entry.get('connector') or '').upper()
    location = entry.get('location')
    if not location or location.startswith('sql:'):
        return None
    if connector == 'DATABRICKS':
        delta = import_provider('mlflow.data.delta_dataset_source', 'mlflow')
        parts = re.findall(r'`((?:[^`]|``)*)`', location)
        head = (entry.get('version') or '').split('_', 1)[0]
        return delta.DeltaDatasetSource(
            delta_table_name='.'.join(part.replace('``', '`')
                                      for part in parts) or location,
            delta_table_version=int(head) if entry.get('version_kind')
            == 'exact' and head.isdigit() else None)
    if connector in _PATH_CONNECTORS:
        registry = import_provider('mlflow.data.dataset_source_registry',
                                   'mlflow')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return registry.resolve_dataset_source(location)
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
def _limited_http() -> Iterator[FrozenSet[str]]:
    r"""MLflow's HTTP limits where unset; yields the names it set."""
    owned = frozenset(name for name in _HTTP_LIMITS if name not in os.environ)
    with environ({name: _HTTP_LIMITS[name] for name in owned}):
        yield owned
