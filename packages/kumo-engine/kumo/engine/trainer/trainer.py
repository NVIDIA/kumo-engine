# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Synchronous local model training, prediction, and bundle loading."""

import functools
import gc
import hashlib
import json
import logging
import os
import shutil
import time
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import pandas as pd
from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.evaluation_counts import (
    at_k_summaries,
    baseline,
    class_names,
    evaluation_summaries,
)
from kumo_core.instrumentation import format_error
from kumo_core.model_plan import ModelPlan as CoreModelPlan
from kumo_core.model_plan import inferred_or_missing_type_to_none
from kumo_core.model_plan_conversion import public_model_plan_from_internal
from kumo_core.pquery import (
    build_generated_training_task_dataset,
    resolve_query_forecast,
)
from kumo_core.summaries import Scalar, Summary
from kumo_core.train.artifact_manifest import PredictionSelectionRef
from kumo_core.training.explain_aggregation import aggregate_subgraph_features
from kumo_core.training.workflow import TrainingWorkflowRequest
from kumo_core.types import ExplainMethod, PredictionArtifactType
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.plugin import (
    BUNDLE_ID_KEY,
    ModelAliasingSink,
    ModelFetchingSink,
    ModelRegistryLocation,
    PublishedModel,
    SourceConnector,
)
from kumoapi.train import TrainingStage
from typing_extensions import Self

from kumo.engine import _deps as dependency
from kumo.engine._data_summaries import (
    snapshot_summaries,
    split_summaries,
    table_details,
    target_edges,
)
from kumo.engine._deps import Dependencies
from kumo.engine._embeddings import (
    no_encoder_inference_warnings,
    require_embedding_models,
    require_text_encoders,
    require_trained_embedding_models,
    with_text_encoders,
)
from kumo.engine._events import Event, EventEmitter, query_sha256
from kumo.engine._manifest import (
    Dependency,
    ModelManifest,
    PredictionManifest,
    PredictionOutput,
    RunScoring,
    TrackingRef,
    artifact_summaries,
    stub,
)
from kumo.engine._mlops import full_error, warn_failure
from kumo.engine._predictor import TRAINER_PREDICTOR
from kumo.engine._scratch import new_run_id, trainer_scratch_dir
from kumo.engine._tracking import (
    RunSpec,
    TrackingSession,
    fit_inputs,
    logged_names,
    system_metrics,
)
from kumo.engine.connector import model_sink
from kumo.engine.connector.base import Connector
from kumo.engine.connector.source_table import SourceTable
from kumo.engine.graph import GraphSnapshot, report_other_graph
from kumo.engine.pquery.prediction_table import PredictionTable
from kumo.engine.pquery.training_table import TrainingTable
from kumo.engine.trainer import _run_manifest
from kumo.engine.trainer._run_spec import FitContext, fit_spec, fitted_spec
from kumo.engine.trainer._scoring import (
    ScoringDefaults,
    normalized_scoring_defaults,
    read_scoring_defaults,
    write_scoring_defaults,
)
from kumo.engine.trainer._spec import (
    HOLDOUT_META_ROLE,
    HOLDOUT_ROLE,
    SPEC_FILE,
    listed_file,
    read_bundle_artifact_paths,
    read_training_spec,
    write_training_spec,
)
from kumo.engine.trainer.output_config import (
    ModelOutputConfig,
    PredictionOutputConfig,
    check_model_location,
)

if TYPE_CHECKING:
    from kumo.engine._monitoring import DriftReport, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    r"""Metrics and on-disk artifacts produced by :meth:`Trainer.fit`.

    `published_model` records the destination's assigned name, version and URI
    when `fit()` exports the bundle. It is `None` without an export.
    `registered_model` is the version a `kumo.MLOps` ``registry`` registered,
    or the registered version this model was loaded from, and
    `registered_alias` the alias it was registered with or loaded by.

    `holdout_path` is the published model's holdout Parquet file written by
    `fit()`, or `None` when it has none (no holdout task, or a model refit on
    the full data). The bundle carries a copy (`holdout.parquet`), so a model
    loaded with `Trainer.load` or `Trainer.load_from_connector` has it too;
    bundles from before this copy was added have none. The file is capped at
    1 GiB in memory; `holdout_truncated` is true when rows were dropped to fit,
    and `None` when the size record could not be read.
    """

    metrics: Dict[str, float]
    artifact_paths: Dict[str, str]
    bundle_dir: Path
    published_model: Optional[PublishedModel] = None
    holdout_path: Optional[Path] = None
    holdout_truncated: Optional[bool] = False
    bundle_id: Optional[str] = None
    registered_model: Optional[PublishedModel] = None
    registered_alias: Optional[str] = None
    scoring_defaults: ScoringDefaults = field(default_factory=ScoringDefaults)
    _trainer: Optional['Trainer'] = field(default=None, repr=False,
                                          compare=False)
    # The fit's `kumo_core.summaries` of its test split, for trackers.
    _summaries: Tuple[Summary, ...] = field(default=(), repr=False,
                                            compare=False)

    @property
    def id(self) -> Optional[str]:
        """The model's id: its bundle id."""
        return self.bundle_id

    @property
    def manifest(self) -> Optional[ModelManifest]:
        """The bundle's ``_provenance.json``; None for older bundles."""
        from kumo.engine.trainer._provenance import read_provenance

        return read_provenance(self.bundle_dir)

    @property
    def snapshot_id(self) -> Optional[str]:
        """The id of the snapshot the model was trained on."""
        if not (self.bundle_dir / SPEC_FILE).is_file():
            return None
        return read_training_spec(self.bundle_dir).snapshot_id

    @property
    def tracking(self) -> Optional[TrackingRef]:
        """The experiment tracker's run of the fit: its ``platform``,
        ``run_id`` and ``url``; None when the fit was not tracked."""
        manifest = self.manifest
        return None if manifest is None else manifest.tracking

    def predict(self, *args: Any, **kwargs: Any) -> 'PredictionResult':
        """Score with this model, as :meth:`Trainer.predict`."""
        return self._model('predict').predict(*args, **kwargs)

    def export(self, config: Any) -> PublishedModel:
        """Publish this model's bundle to a ``ModelOutputConfig``, or
        register it at a `kumo.MLOps` ``registry`` through its tracker; a
        registry that already holds the bundle returns that version."""
        from kumo.engine._mlops import MLOps

        if isinstance(config, MLOps):
            return _register_result(self, config)
        _require_publisher(config)
        return _publish(self.bundle_dir, self.bundle_id, self.metrics, config)

    def with_scoring_defaults(self,
                              defaults: ScoringDefaults) -> 'TrainingResult':
        """A copy of this model whose ``predict`` applies ``defaults`` when a
        call omits them; a new bundle, whose manifest names this one in
        ``parents.derived_from``. To register a model with its defaults,
        pass ``fit(scoring_defaults=...)`` instead."""
        from kumo_core.artifacts import bundle_task_type

        from kumo.engine.trainer._provenance import record_derivation

        if self._trainer is None:
            raise RuntimeError('This result has no model; use Trainer.load on '
                               'its bundle_dir.')
        defaults = normalized_scoring_defaults(
            defaults, bundle_task_type(self.artifact_paths))
        bundle_dir = trainer_scratch_dir(new_run_id()) / 'derived-model'
        shutil.copytree(self.bundle_dir, bundle_dir,
                        copy_function=_link_or_copy)
        write_scoring_defaults(bundle_dir, defaults)
        try:
            record_derivation(bundle_dir, self.bundle_dir, self.bundle_id)
        except Exception as error:
            warnings.warn(
                f"Could not record the parent model in _provenance.json: "
                f"{error}", UserWarning, stacklevel=2)
        result = Trainer._load_bundle(bundle_dir,
                                      self._trainer._embedding_models).result
        assert result is not None
        return result

    def evaluate(self, *args: Any, **kwargs: Any) -> 'EvaluationResult':
        """Evaluate a run of this model, as :meth:`Trainer.evaluate`."""
        return self._model('evaluate').evaluate(*args, **kwargs)

    def score_drift(self, *args: Any, **kwargs: Any) -> 'DriftReport':
        """Score drift of a run of this model, as :meth:`Trainer.score_drift`."""
        return self._model('score_drift').score_drift(*args, **kwargs)

    def _model(self, call: str) -> 'Trainer':
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError(f'This result has no model to {call} with; '
                               'use Trainer.load on its bundle_dir.')
        current = trainer._training_result
        if current is None or current.bundle_dir != self.bundle_dir:
            trainer = Trainer._load_bundle(self.bundle_dir,
                                           trainer._embedding_models,
                                           accelerator=trainer._accelerator)
            self._trainer = trainer
        return trainer

    def holdout_evaluation(
        self, binary_classification_threshold: Optional[float] = None
    ) -> 'EvaluationResult':
        r"""The holdout as evaluated rows, with the columns and types
        `Trainer.evaluate` writes: export it as a warehouse monitor's
        baseline. ``TARGET_PRED`` uses the threshold passed, else the stored
        default, else is null, as ``predict`` does."""
        from kumo.engine._monitoring import holdout_evaluation

        if binary_classification_threshold is None:
            binary_classification_threshold = (
                self.scoring_defaults.binary_classification_threshold)
        return holdout_evaluation(self, binary_classification_threshold)

    def holdout_df(self) -> pd.DataFrame:
        r"""Read the holdout predictions as a pandas DataFrame.

        Raises `FileNotFoundError` if no holdout was produced. Reads the whole
        file into memory; check `holdout_truncated` for dropped rows.
        """
        if self.holdout_path is None:
            raise FileNotFoundError(
                'No holdout dataframe was produced for this result')
        if self.holdout_truncated:
            logger.warning('Holdout dataframe was truncated to fit the size '
                           'cap; it does not contain every holdout row.')
        elif self.holdout_truncated is None:
            logger.warning('Holdout truncation is unknown; it may not contain '
                           'every holdout row.')
        return pd.read_parquet(self.holdout_path)


@dataclass(frozen=True)
class PredictionResult:
    r"""Durable prediction output. Rows stay on disk until ``frame`` is read.

    ``num_rows`` reads bounded producer metadata; ``files`` resolves selected
    Parquet files. Accessing ``frame`` explicitly collects the whole result
    into memory and is intended only for outputs that fit in memory.
    ``id`` names the run; ``manifest`` is its record (None if it could not
    be written), from which ``bundle_id`` and ``snapshot_id`` are read.
    """

    selection: Optional[PredictionSelectionRef]
    artifact_type: PredictionArtifactType
    _exported: Optional['_Export'] = field(default=None, compare=False,
                                           repr=False)

    @classmethod
    def load_from_connector(cls, connector: Connector, *,
                            path: str) -> 'PredictionResult':
        r"""Load a Parquet directory written by :meth:`export` for monitoring.

        A `kumo.MLOps` ``connector`` reads ``path`` under its artifact root,
        such as ``predictions/<prediction id>``; a run directory loads its
        ``predictions/``. The export must be complete (its ``_SUCCESS``
        marker present). Only
        its ``_provenance.json`` is fetched here; the Parquet parts are
        downloaded, in parallel, when ``files`` is first read, into a local
        directory removed when the result is garbage-collected.
        """
        import tempfile
        import weakref

        from pyarrow.fs import FileType

        from kumo.engine._manifest import read_manifest
        from kumo.engine._mlops import on_root
        from kumo.engine._scratch import get_scratch_dir
        from kumo.engine.connector import file_store

        connector, path = on_root(connector, path)
        store = file_store(connector, path)
        if store is None:
            raise TypeError(f'{path} is not a file destination of this '
                            f'connector')
        filesystem, root = store
        root = root.rstrip('/')
        if (filesystem.get_file_info(f'{root}/_SUCCESS').type
                == FileType.NotFound and filesystem.get_file_info(
                    f'{root}/predictions/_SUCCESS').type == FileType.File):
            root, path = f'{root}/predictions', f"{path.rstrip('/')}/predictions"
        if filesystem.get_file_info(
                f'{root}/_SUCCESS').type == FileType.NotFound:
            raise FileNotFoundError(
                f'{path} has no _SUCCESS marker: the export is incomplete or '
                f'still running')
        scratch = get_scratch_dir()
        scratch.mkdir(parents=True, exist_ok=True)
        local = Path(tempfile.mkdtemp(prefix='fetched-predictions-',
                                      dir=scratch))
        try:
            with filesystem.open_input_stream(
                    f'{root}/{PROVENANCE_FILE}') as reader:
                (local / PROVENANCE_FILE).write_bytes(reader.read())
        except (FileNotFoundError, OSError):
            pass
        manifest = read_manifest(local / PROVENANCE_FILE, PredictionManifest)
        here = str(_run_manifest.destination_record(connector,
                                                    path).location).rstrip('/')
        kind = next((PredictionArtifactType(record.artifact.upper())
                     for record in (manifest and manifest.outputs or [])
                     if str(record.destination.location).rstrip('/') == here),
                    PredictionArtifactType.EMBEDDINGS
                    if root.rsplit('/', 1)[-1] == 'embeddings' else
                    PredictionArtifactType.PREDICTIONS)
        exported = _Export(filesystem, root, local, kind)
        weakref.finalize(exported, shutil.rmtree, str(local), True)
        return cls(None, kind, _exported=exported)

    @property
    def id(self) -> Optional[str]:
        if self.selection is not None:
            return self.selection.job_id
        manifest = self.manifest
        return None if manifest is None else manifest.id

    @property
    def _manifest_path(self) -> Path:
        if self._exported is not None:
            return self._exported.local / PROVENANCE_FILE
        return Path(self.selection.pointer_path).parent / PROVENANCE_FILE

    @property
    def manifest(self) -> Optional[PredictionManifest]:
        from kumo.engine._manifest import read_manifest

        return read_manifest(self._manifest_path, PredictionManifest)

    @property
    def bundle_id(self) -> Optional[str]:
        manifest = self.manifest
        return None if manifest is None else manifest.parents.model.id

    @property
    def snapshot_id(self) -> Optional[str]:
        manifest = self.manifest
        return None if manifest is None else manifest.parents.snapshot.id

    @property
    def requested_rows(self) -> Optional[int]:
        r"""Entities asked for, from the run's manifest."""
        return self._table_count('requested_rows')

    @property
    def scored_rows(self) -> Optional[int]:
        r"""Entities scored, from the run's manifest."""
        return self._table_count('scored_rows')

    @property
    def skipped_rows(self) -> Optional[int]:
        r"""Requested entities that were not scored (see the manifest's
        ``prediction_table.skip_reason``), from the run's manifest."""
        return self._table_count('skipped_rows')

    def _table_count(self, name: str) -> Optional[int]:
        manifest = self.manifest
        return None if manifest is None else getattr(
            manifest.prediction_table, name)

    @property
    def num_rows(self) -> int:
        if self._exported is not None:
            from pyarrow import parquet
            return sum(
                parquet.ParquetFile(path).metadata.num_rows
                for path in self.files)
        from kumo_core.train import artifact_manifest
        selected = artifact_manifest.load_prediction_selection(
            self.selection, validate_objects=False)
        return next(item.num_rows for item in selected.artifacts
                    if item.artifact_type is self.artifact_type)

    @property
    def files(self) -> tuple[str, ...]:
        from kumo_core.train import artifact_manifest
        if self._exported is not None:
            if self._exported.kind is not self.artifact_type:
                raise ValueError(
                    f'This export holds {self._exported.kind.value.lower()}')
            return self._exported.files()
        resolved = artifact_manifest.resolve_prediction_selection_artifact(
            self.selection, self.artifact_type)
        if resolved is None:
            raise ValueError(
                f'This prediction run did not generate {self.artifact_type.value.lower()}'
            )
        assert resolved.files is not None
        return resolved.files

    @property
    def frame(self) -> pd.DataFrame:
        return pd.concat((pd.read_parquet(path) for path in self.files),
                         ignore_index=True)

    def export(self, output_config: PredictionOutputConfig) -> None:
        if output_config.extra_fields:
            raise ValueError(
                'extra_fields are chosen at prediction time and cannot be '
                'applied to an existing result; pass them to '
                'Trainer.predict(output_config=...)')
        self._export(output_config, self.manifest)

    def _export(self, output_config: PredictionOutputConfig,
                sidecar: Optional[PredictionManifest] = None,
                records: Optional[List[PredictionOutput]] = None,
                written: Optional[List[PredictionOutput]] = None) -> None:
        for destination in output_config.destinations().values():
            output_config.writer(destination)
        outputs = []
        for name, destination in output_config._targets(self.id).items():
            kind = PredictionArtifactType(name.upper())
            files = replace(self, artifact_type=kind).files
            outputs.append((kind, destination, files))
        payload = _sidecar_bytes(sidecar)
        bundle_id = None if sidecar is None else sidecar.parents.model.id
        for index, (kind, destination, files) in enumerate(outputs):
            extra: Dict[str, Any] = ({'sidecars': {PROVENANCE_FILE: payload}}
                                     if payload is not None and _holds_parquet(
                                         output_config.connector, destination)
                                     else {})
            columns = output_config._id_columns(destination, self.id,
                                                bundle_id)
            if columns:
                extra['extra_columns'] = columns
            output_config._write(destination, list(files),
                                 export_id=f'{self.id}-{kind.value}', **extra)
            if records is not None and written is not None:
                written.append(records[index])


@dataclass(frozen=True)
class _Export:
    filesystem: Any
    root: str
    local: Path
    kind: PredictionArtifactType

    def files(self) -> tuple[str, ...]:
        parts = self.local / 'parts'
        if not parts.is_dir():
            from kumo_core.sources.transfer import StagedFile, fetch_all
            from pyarrow.fs import FileSelector, FileType

            def fetch(staged: StagedFile, target: str) -> None:
                with self.filesystem.open_input_stream(staged.path) as reader, \
                        open(target, 'wb') as writer:
                    shutil.copyfileobj(reader, writer, 1 << 20)

            partial = self.local / 'parts.partial'
            shutil.rmtree(partial, ignore_errors=True)
            listed = self.filesystem.get_file_info(FileSelector(self.root))
            fetch_all([
                StagedFile(info.path, info.size) for info in listed
                if info.type == FileType.File and info.path.endswith(
                    '.parquet')
            ], str(partial), fetch=fetch)
            partial.rename(parts)
        return tuple(sorted(map(str, parts.glob('*.parquet'))))


def _sidecar_bytes(manifest: Optional[PredictionManifest]) -> Optional[bytes]:
    from kumo.engine._manifest import manifest_bytes

    try:
        return None if manifest is None else manifest_bytes(manifest)
    except Exception as error:
        warnings.warn(f"Exporting without {PROVENANCE_FILE}: {error}",
                      UserWarning, stacklevel=4)
        return None


def _holds_parquet(connector: Connector, destination: str) -> bool:
    from kumo_core.sources import layout

    from kumo.engine.connector import is_file_destination

    return (is_file_destination(connector, destination)
            and layout.file_type(destination.rstrip('/')) != layout.CSV)


def _find_holdout(log_dir: str) -> tuple[Optional[Path], Optional[bool]]:
    from kumo_core.train.output import EvalOutput
    path = Path(EvalOutput.get_df_path(log_dir, TrainingStage.TEST))
    if not path.is_file():
        return None, False
    meta = Path(EvalOutput.get_df_meta_path(log_dir, TrainingStage.TEST))
    try:
        truncated = bool(json.loads(meta.read_text())['truncated'])
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning(
            'Holdout size record %s is missing or unreadable; '
            'truncation is unknown.', meta)
        truncated = None
    return path, truncated


HOLDOUT_FILE = 'holdout.parquet'
HOLDOUT_META_FILE = 'holdout-meta.json'


def _bundle_holdout(result: TrainingResult) -> TrainingResult:
    r"""Copy the holdout and its size record into the bundle, so a model
    loaded from a registry or path still has them."""
    from kumo_core.artifacts import add_bundle_file

    if result.holdout_path is None:
        return result
    try:
        shutil.copyfile(result.holdout_path, result.bundle_dir / HOLDOUT_FILE)
        add_bundle_file(result.bundle_dir, HOLDOUT_ROLE, HOLDOUT_FILE)
        if result.holdout_truncated is not None:
            (result.bundle_dir / HOLDOUT_META_FILE).write_text(
                json.dumps({'truncated': result.holdout_truncated}))
            add_bundle_file(result.bundle_dir, HOLDOUT_META_ROLE,
                            HOLDOUT_META_FILE)
    except Exception as error:
        warnings.warn(
            f"Could not add the holdout to the model bundle: {error}; a "
            f"loaded copy of this model has no holdout_df().", UserWarning,
            stacklevel=4)
    return result


def _listed_holdout(bundle_dir: Path) -> tuple[Optional[Path], Optional[bool]]:
    path = listed_file(bundle_dir, HOLDOUT_ROLE)
    if path is None or not path.is_file():
        return None, False
    meta = listed_file(bundle_dir, HOLDOUT_META_ROLE)
    try:
        truncated = (None if meta is None else bool(
            json.loads(meta.read_text())['truncated']))
    except (OSError, ValueError, KeyError, TypeError):
        truncated = None
    return path, truncated


def _store_and_register(
    bundle_dir: Path,
    manifest: Optional[ModelManifest],
    settings: Any,
    *,
    session: Optional[TrackingSession] = None,
    snapshot: Any = None,
    metrics: Mapping[str, float] = {},
    warn: bool = True,
    stored: Optional[BaseException] = None,
) -> Tuple[Optional[PublishedModel], Optional[BaseException]]:
    r"""Store a fitted bundle under the artifact root, log it in the
    session's run and register it, as MLOps ``settings`` say; returns the
    registered version and the first failure. A model whose bundle or
    assets (``stored``, the asset failure) could not be stored is not
    registered."""
    if stored is None and settings is not None and settings.wants('model'):
        try:
            _store_model(bundle_dir, manifest, settings)
        except Exception as error:
            stored = error
    if stored is not None and warn:
        warn_failure(f'{stored}; the model is not registered', stored, 3)
    registered, error = (None, None) if session is None else session.log_model(
        bundle_dir, manifest, snapshot,
        None if stored else _mlflow_registry(settings))
    registered = _with_metrics(registered, metrics)
    error = stored or error
    registry = None if stored else _connector_registry(settings)
    if registry is None:
        return registered, error
    errors: List[BaseException] = []
    try:
        registered = _publish(bundle_dir, None if manifest is None else
                              manifest.id, metrics, registry, errors, warn)
    except Exception as failure:
        if warn:
            warn_failure(
                f"Could not register the model: {full_error(failure)}",
                failure, 3)
        return registered, error or failure
    return registered, error or (errors or [None])[0]


def _store_model(bundle_dir: Path, manifest: Optional[ModelManifest],
                 settings: Any) -> None:
    r"""Copy the bundle to ``<root>/models/<bundle id>/`` unless a complete
    copy is there."""
    from kumo_core.artifacts import PROVENANCE_FILE, add_bundle_file
    from kumo_core.object_store_sink import holds_bundle, upload_bundle
    from pyarrow.fs import FileSelector

    from kumo.engine.connector import file_store
    from kumo.engine.trainer._spec import PROVENANCE_ROLE

    if manifest is None:
        return
    try:
        if settings.locate(bundle_dir / PROVENANCE_FILE, ModelManifest,
                           f'models/{manifest.id}') is not None:
            add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)
        filesystem, root = file_store(
            settings.root.connector,
            settings.root.location(f'models/{manifest.id}'))
        if holds_bundle(filesystem, root, str(bundle_dir)):
            return
        if filesystem.get_file_info(
                FileSelector(root, allow_not_found=True)):
            filesystem.delete_dir_contents(root, missing_dir_ok=True)
        upload_bundle(filesystem, root, str(bundle_dir))
    except Exception as error:
        raise RuntimeError(f"Could not store model {manifest.id} under the "
                           f"artifact root: {error}") from error


def _published_holdout(trainer: Any) -> tuple[Optional[Path], Optional[bool]]:
    from kumo_core.training.config import FitMode

    published = trainer.published_fit_mode()
    if published is FitMode.refit_full:
        logger.info('The published model is refit on the full data, including '
                    'the test split, so it has no holdout predictions.')
        return None, False
    return _find_holdout(trainer.get_log_dir(fit_mode=published))


def _option_source(argument: Any, requested: Any, value: Any,
                   fallback: str) -> str:
    if value is None:
        return 'none'
    if argument is not None:
        return 'argument'
    return 'bundle' if requested is not None else fallback


def _collect_garbage(stage: str) -> None:
    # Lightning links each trainer to its data module and model in reference
    # cycles; free a finished run now rather than at a later collection.
    started = time.perf_counter()
    collected = gc.collect()
    logger.debug('%s: garbage collection collected %d objects in %.2fs', stage,
                 collected,
                 time.perf_counter() - started)


def _to_core_model_plan(model_plan: ModelPlan) -> CoreModelPlan:
    """Convert a public model plan to the internal workflow representation."""
    from kumo_core.model_plan_conversion import internal_model_plan_from_public

    return internal_model_plan_from_public(model_plan)


def _link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _require_publisher(export_config: ModelOutputConfig) -> None:
    if not callable(
            getattr(model_sink(export_config.connector, export_config.path),
                    'publish_model', None)):
        raise NotImplementedError(
            f'{type(export_config.connector).__name__} cannot publish '
            f'models')
    check_model_location(export_config)


def _register_result(result: TrainingResult, settings: Any) -> PublishedModel:
    if settings.registry is None:
        raise ValueError('export(cfg) registers at cfg.registry, which is '
                         'not set.')
    register = getattr(settings.tracking, 'register', None)
    if callable(register):
        return cast(PublishedModel, _with_metrics(
            register(result, settings.registry), result.metrics))
    registry = cast(ModelOutputConfig, _connector_registry(settings))
    _require_publisher(registry)
    return _publish(result.bundle_dir, result.bundle_id, result.metrics,
                    registry)


def _unlogged(summaries: Sequence[Summary],
              metrics: Mapping[str, float]) -> Tuple[Summary, ...]:
    r"""``summaries`` without the scalars ``metrics`` already logs, which
    stay the trainer's own."""
    logged = set(logged_names(metrics))
    return tuple(summary for summary in summaries
                 if not (isinstance(summary, Scalar) and summary.name in logged))


def _with_metrics(model: Optional[PublishedModel],
                  metrics: Mapping[str, float]) -> Optional[PublishedModel]:
    r"""``model`` with the fit's metrics, where its registry returned none
    (an MLflow version, unlike a Snowflake one, carries none)."""
    if model is None or model.metrics:
        return model
    return replace(model, metrics=dict(metrics))


def _mlflow_registry(settings: Any) -> Optional[ModelRegistryLocation]:
    if settings is None or not _logs_models(settings.tracking):
        return None
    return settings.registry


def _connector_registry(settings: Any) -> Optional[ModelOutputConfig]:
    if (settings is None or settings.registry is None
            or _logs_models(settings.tracking)):
        return None
    return ModelOutputConfig(settings.tracking.connector,
                             location=settings.registry)


def _logs_models(tracking: Any) -> bool:
    return callable(getattr(tracking, 'log_model', None))


def _publish(bundle_dir: Path, bundle_id: Optional[str],
             metrics: Mapping[str, float], export_config: ModelOutputConfig,
             errors: Optional[List[BaseException]] = None,
             warn: bool = True) -> PublishedModel:
    metadata = {}
    if (bundle_dir / SPEC_FILE).is_file():
        spec = read_training_spec(bundle_dir)
        metadata = {
            name: str(getattr(spec, name))
            for name in ('spec_version', 'predictive_query', 'snapshot_id')
        }
    metadata.update(registry_pointers(bundle_id))
    location = export_config.location
    alias = location.alias if location else None
    if location and alias:
        location = replace(location, alias=None)
    sink = model_sink(export_config.connector, export_config.path)
    published = sink.publish_model(
        export_config.connector,
        str(bundle_dir),
        path=export_config.path,
        location=location,
        metadata=metadata,
        metrics=metrics,
    )
    if location and alias:
        try:
            _set_alias(export_config.connector,
                       replace(location, version=published.version), alias)
        except Exception as error:
            if warn:
                warn_failure(
                    f"Published {published.name} version "
                    f"{published.version} without alias {alias}: "
                    f"{full_error(error)}", error, 4)
            if errors is not None:
                errors.append(error)
    return published


def _stage_model_bundle(artifacts: Mapping[str, str], dest: Path) -> Path:
    from kumo_core.artifacts import stage_model_bundle
    return stage_model_bundle(artifacts, dest)


_F = TypeVar('_F', bound=Callable[..., Any])


def _without_torch_scatter_hint(method: _F) -> _F:
    r"""Runs ``method`` without PyTorch Geometric's GPU hint that min/max
    aggregation "can be accelerated via the 'torch-scatter' package",
    Lightning's "Tip" log lines advertising its cloud products, Lightning's
    note that the loader's IterableDataset defines ``__len__``, and Python's
    note that forking a multi-threaded process may deadlock (the loader
    workers fork before any thread that matters holds a lock).

    kumo-engine uses PyTorch's own scatter instead of that optional extension
    (see ``test_wheel_uses_torch_fallbacks_without_native_extensions``), so
    the hint, which PyG raises for every such aggregation, is not actionable.
    """
    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tips = logging.getLogger('pytorch_lightning.utilities.rank_zero')
        tips.addFilter(_no_lightning_tip)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message=r"The usage of `scatter\(reduce='\w+'\)` can be "
                    r"accelerated via the 'torch-scatter' package",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    'ignore', message=r'Your `IterableDataset` has `__len__`',
                    category=UserWarning)
                warnings.filterwarnings(
                    'ignore', message=r'This process .* is multi-threaded, '
                    r'use of fork\(\) may lead to deadlocks',
                    category=DeprecationWarning)
                return method(*args, **kwargs)
        finally:
            tips.removeFilter(_no_lightning_tip)

    return cast(_F, wrapper)


def _no_lightning_tip(record: logging.LogRecord) -> bool:
    return not str(record.msg).startswith('💡 Tip')


class Trainer:
    r"""Trains and runs a GNN model in-process.

    Args:
        model_plan: shared, backend-agnostic training config
            (`kumoapi.model_plan.ModelPlan`) -- identical to what the cloud
            pipeline consumes, which is what keeps ML behavior consistent
            between hosted and local training.
        embedding_models: the word-embedding models for text columns, by
            name. Defaults to the installed ones (``kumo-engine[text]``).
        accelerator: Lightning device selection: ``auto`` (default), ``cpu``,
            ``gpu``, or ``mps``. Applies to training and prediction.
    """
    def __init__(
        self,
        model_plan: ModelPlan,
        embedding_models: Optional[EmbeddingModelStore] = None,
        *,
        accelerator: Literal['cpu', 'gpu', 'mps', 'auto'] = 'auto',
    ):
        if accelerator not in ('cpu', 'gpu', 'mps', 'auto'):
            raise ValueError(
                "accelerator must be one of 'cpu', 'gpu', 'mps', or 'auto'")
        self._accelerator = accelerator
        self._model_plan = model_plan
        self._embedding_models = (embedding_models if embedding_models
                                  is not None else EmbeddingModelStore())
        self._id = new_run_id()
        self._training_result: Optional[TrainingResult] = None
        self._snapshot_id: Optional[str] = None
        self._graph_config: Optional[Dict[str, Any]] = None
        self._training_query: Optional[str] = None
        self._training_table_generation_plan: Optional[
            TrainingTableGenerationPlan] = None

    @property
    def model_plan(self) -> ModelPlan:
        """Training configuration used by the next fit call."""
        return self._model_plan

    @model_plan.setter
    def model_plan(self, model_plan: ModelPlan) -> None:
        self._model_plan = model_plan

    @property
    def training_table_generation_plan(
            self) -> Optional[TrainingTableGenerationPlan]:
        """Effective table-generation settings saved by the last fit/load."""
        return self._training_table_generation_plan

    @property
    def scratch_dir(self) -> Path:
        return trainer_scratch_dir(self._id)

    @property
    def result(self) -> Optional[TrainingResult]:
        """The last fit's result, or the loaded bundle's."""
        return self._training_result

    def _fitted(self, method: str) -> TrainingResult:
        if self._training_result is None:
            raise RuntimeError(
                f"Trainer.{method}() requires a prior Trainer.fit() call on "
                "this same Trainer instance.")
        return self._training_result

    def export(self, config: Any) -> PublishedModel:
        """Publish the fitted or loaded model bundle, as
        :meth:`TrainingResult.export`."""
        return self._fitted('export').export(config)

    @_without_torch_scatter_hint
    def fit(
        self,
        train_table: TrainingTable,
        *,
        snapshot: GraphSnapshot,
        export_config: Optional[ModelOutputConfig] = None,
        scoring_defaults: Optional[ScoringDefaults] = None,
        warm_start: Optional[Self] = None,
        mlops: Optional[Any] = None,
        callbacks: Sequence[Callable[[Event], None]] = (),
    ) -> TrainingResult:
        r"""Train a model against `train_table`.

        The table supplies rows, labels, splits, and generation settings.
        ``snapshot`` independently supplies the graph data, topology, and
        feature schema. Fitting consumes the table without regenerating it.

        Unless ``model_plan.training_job.disable_explain`` is set, supported
        tasks also produce gradient scores, column analysis, and subgraph
        explanations under the selected trainer's ``explain_test/``,
        ``column_analysis/test/``, and ``subgraph_collector/test/``
        directories in this run. These diagnostics are not part of the
        published model bundle. Explanation failures are logged without
        discarding the successfully trained model.

        Args:
            train_table: a table produced by `PredictiveQuery.
                generate_training_table(snapshot)`.
            snapshot: graph data and features to train against.
            export_config: if given, publishes the trained model bundle.
            scoring_defaults: prediction arguments the bundle applies when
                ``predict`` omits them, such as a classification threshold;
                stored, registered and served with the model.
            warm_start: a fitted trainer or one restored with ``load`` or
                ``load_from_connector``. Reuses its selected weights, optimizer
                state, fitted encoders, architecture and neighbor sampling;
                this trainer's training controls still apply. Task types must
                match. Forecasting warm starts are unsupported. Older bundles
                may contain an AutoML search space instead of the fitted
                architecture; they support prediction but are unreliable
                warm-start sources.
            mlops: a `kumo.MLOps`; with MLOps on (this or a `kumo.mlops`
                block), the bundle is also stored under the artifact root and
                the fit is recorded by its ``tracking``.
            callbacks: called with each progress
                `kumo.engine.tracking.Event`; a callback that raises is logged
                and not called again.

        Each call returns a new `TrainingResult`; `Trainer.result` holds the
        latest.
        """
        if export_config is not None:
            _require_publisher(export_config)
        from kumo.engine._mlops import raise_on_error, resolve

        settings = resolve(mlops)
        registry = _connector_registry(settings)
        if registry is not None:
            _require_publisher(registry)
        run_id = f'{self._id}-fit-{new_run_id()}'
        warn = settings is None or settings.on_error == 'warn'
        session = TrackingSession(
            None if settings is None else settings.tracking, warn=warn)
        events = EventEmitter(
            [*callbacks, session.on_event] if session.enabled else callbacks)
        deps = Dependencies(settings=settings)
        fit_start = time.monotonic()
        try:
            started = {
                'snapshot_id': snapshot.id,
                'query_sha256': query_sha256(train_table.query),
            }
        except Exception:
            started = {}
        events.emit('fit_started', params=started)
        try:
            result = self._fit(train_table, snapshot=snapshot,
                               export_config=export_config,
                               scoring_defaults=scoring_defaults,
                               warm_start=warm_start, run_id=run_id,
                               events=events, tracking=session,
                               deps=deps, model_name=None if (
                                   settings is None or settings.registry is
                                   None) else settings.registry.model_name)
        except BaseException as error:
            events.emit('fit_ended', params={
                'status': 'failed',
                'error': format_error(error),
            })
            session.finish(None, error)
            raise
        seconds = time.monotonic() - fit_start
        events.emit('fit_ended', metrics={
            **result.metrics, 'seconds': seconds
        }, params={'status': 'succeeded'})
        alias = None if settings is None or settings.registry is None else (
            settings.registry.alias)
        registered, error = _store_and_register(
            result.bundle_dir, result.manifest or (
                None if result.bundle_id is None else stub(
                    'model', result.bundle_id)), settings, session=session,
            snapshot=snapshot, metrics=result.metrics, warn=warn,
            stored=(deps.errors or [None])[0])
        if registered is not None:
            result = replace(result, registered_model=registered,
                             registered_alias=alias)
            self._training_result = result
        session.finish(
            lambda: fitted_spec(
                cast(RunSpec, session.spec), result, seconds, (
                    *result._summaries,
                    *snapshot_summaries(Path(snapshot.index_path),
                                        snapshot.graph_config),
                    *([] if result.manifest is None else artifact_summaries(
                        result.manifest, 'training', settings,
                        table_details(Path(snapshot.index_path)))),
                    *split_summaries(train_table.staging_dir))), error)
        raise_on_error(settings, session.error or error,
                       'Trainer.fit (its model is in trainer.result)')
        return result

    def _fit(
        self,
        train_table: TrainingTable,
        *,
        snapshot: GraphSnapshot,
        export_config: Optional[ModelOutputConfig],
        scoring_defaults: Optional[ScoringDefaults],
        warm_start: Optional[Self],
        run_id: str,
        events: EventEmitter,
        tracking: TrackingSession,
        deps: Dependencies,
        model_name: Optional[str] = None,
    ) -> TrainingResult:
        from kumo_core.automl.runtime import (
            inherit_warm_start_model_plan,
            validate_warm_start_task_type,
        )
        from kumo_core.io.json import load_json, to_json_dict
        from kumo_core.training.orchestration import (
            CallableTrainingOptions,
            ResolvedTrainingInput,
            run_callable_training,
        )
        from kumo_core.training.search import AutoMLSearchOptions
        from kumo_core.training.table import GeneratedTrainTableResult
        from kumo_core.training.workflow import (
            PreparedGraph,
            materialize_generated_training_task,
            resolve_model_plan,
        )
        from kumo_core.types import TaskType

        from kumo.engine.trainer._provenance import parent_model
        from kumo.engine.trainer._runtime import local_training_bindings

        started_at = datetime.now(timezone.utc)
        run_dir = self.scratch_dir / 'runs' / run_id

        model_plan = _to_core_model_plan(self._model_plan)
        warm_start_artifacts = None
        warm_start_bundle_id = None
        effective_plan = self._model_plan
        if warm_start is not None:
            if warm_start._training_result is None:
                raise ValueError(
                    "warm_start requires a fitted or loaded Trainer")
            warm_start_artifacts, warm_start_bundle_id = (
                read_bundle_artifact_paths(
                    warm_start._training_result.bundle_dir))
            inherit_warm_start_model_plan(
                model_plan,
                load_json(warm_start_artifacts['resolved_model_plan'],
                          CoreModelPlan),
            )
            effective_plan = public_model_plan_from_internal(model_plan)
        workflow = TrainingWorkflowRequest.from_spec(
            graph=snapshot.graph_config,
            predictive_query=train_table.query,
            prepared_graph=PreparedGraph(
                index_id=snapshot.id,
                index_path=str(snapshot.index_path),
            ),
            run_id=run_id,
            scratch_dir=str(self.scratch_dir),
            artifact_dir=str(run_dir),
            model_plan=model_plan,
            training_table_generation_plan=train_table.generation_plan,
            search=AutoMLSearchOptions(),
        )
        if scoring_defaults is not None:
            scoring_defaults = normalized_scoring_defaults(
                scoring_defaults, TaskType(workflow.query.task_type))
        if warm_start_artifacts is not None:
            from kumo_core.artifacts import bundle_task_type

            validate_warm_start_task_type(
                TaskType(workflow.query.task_type),
                bundle_task_type(warm_start_artifacts),
            )
        from kumo_core.pquery import (
            query_is_temporal,
            target_to_entity_paths,
            to_plan_graph,
        )
        from kumo_core.training.custom_table import contract
        from kumo_core.training.export import read_spec, stage_parts
        from kumoapi.pquery import QueryType
        from pyarrow import parquet

        parts = stage_parts(read_spec(str(train_table.staging_dir)))
        seed_part = next(part for paths in parts.values() for part in paths)
        seed_schema = parquet.read_schema(train_table.staging_dir / seed_part)
        entity_tables = [workflow.query.entity_table]
        if (workflow.query.rhs_node_type is not None
                and workflow.query.rhs_node_type not in entity_tables):
            entity_tables.append(workflow.query.rhs_node_type)
        # Validate a copy with query/schema facts before extending the index.
        # Materialization still resolves defaults and data-derived fanouts.
        resolve_model_plan(
            model_plan,
            workflow.search,
            training_table_generation_plan=train_table.generation_plan,
            task_type=TaskType(workflow.query.task_type),
            query_type=QueryType(workflow.query.query_type),
            has_train_table_time_col=query_is_temporal(workflow.query,
                                                       workflow.graph),
            entity_table_names=entity_tables,
            target_to_entity_paths=target_to_entity_paths(
                workflow.query, workflow.graph),
            has_train_table_weight_col=contract()['weight_output_column']
            in seed_schema.names,
            graph=to_plan_graph(workflow.graph),
        )
        ingest_tt_id = f'tt-{run_id}'
        lag_timesteps = inferred_or_missing_type_to_none(
            train_table.generation_plan.lag_timesteps)
        year_over_year = inferred_or_missing_type_to_none(
            train_table.generation_plan.year_over_year)
        with no_encoder_inference_warnings(
                effective_plan.column_processing.encoder_overrides or {}):
            task_dataset = build_generated_training_task_dataset(
                query=workflow.query,
                graph=workflow.graph,
                graph_index_path=workflow.prepared_graph.index_path,
                run_id=run_id,
                artifact_dir=str(run_dir),
                tt_id=ingest_tt_id,
                required_stages=train_table.required_stages,
                lag_timesteps=(None if lag_timesteps is None else
                               int(lag_timesteps)),
                year_over_year=bool(year_over_year),
            )
        require_embedding_models(
            self._embedding_models, task_dataset.feature_schema,
            effective_plan.column_processing.encoder_overrides)
        run_dir.mkdir(parents=True, exist_ok=True)
        materialized = materialize_generated_training_task(
            workflow,
            task_dataset,
            GeneratedTrainTableResult(
                train_table_path=str(train_table.train_table_path),
                manifest_path=str(train_table.manifest_path),
                staging_dir=str(train_table.staging_dir),
            ),
            tt_id=ingest_tt_id,
        )
        del task_dataset
        if (materialized.task.split.stage_counts.get('test') == 0
                and not materialized.task.task_type.is_link_pred):
            raise ValueError('The test split is empty: shorten the query '
                             'window or add data.')
        tracking.start(lambda: fit_spec(FitContext(
            query=train_table.query,
            model_plan=to_json_dict(materialized.model_plan),
            inputs=fit_inputs(train_table.manifest),
            started=started_at,
            snapshot_id=snapshot.id,
            task_type=materialized.task.task_type.value,
            model_name=model_name,
        )))
        training_input = ResolvedTrainingInput(
            task=materialized.task,
            model_plan=materialized.model_plan,
            home_dir=str(run_dir),
            task_dataset_bundle_dir=materialized.bundle_dir,
            warm_start_artifacts=warm_start_artifacts,
        )
        threshold = None if scoring_defaults is None else (
            scoring_defaults.binary_classification_threshold)
        result = run_callable_training(
            training_input,
            bindings=local_training_bindings(
                training_input, run_name=run_id,
                embedding_models=self._embedding_models,
                accelerator=self._accelerator, instrumentation=events),
            options=CallableTrainingOptions(
                binary_classification_threshold=threshold,
                target_edges=target_edges(train_table.staging_dir)),
        )

        if not result.explain_disabled:
            try:
                # The shared trainer owns task support, checkpoint selection
                # (refit_trainval when available), and artifact generation.
                result.best_trainer.explain(
                    explain_methods=[
                        ExplainMethod.COLUMN_ANALYSIS,
                        ExplainMethod.GRADIENT,
                        ExplainMethod.SUBGRAPH_COLLECTOR,
                    ],
                    stage='test',
                )
                aggregate_subgraph_features(
                    str(
                        Path(result.best_trainer.subgraph_collector_dir()) /
                        'test'), TrainingStage.TEST)
            except Exception:
                logger.exception(
                    'Training succeeded, but post-training explanations '
                    'failed for run %s. The trained model remains available.',
                    run_id)

        holdout_path, holdout_truncated = _published_holdout(
            result.best_trainer)
        fit_mode = result.best_trainer.published_fit_mode()
        fit_mode = None if fit_mode is None else fit_mode.value
        task_type = TaskType(workflow.query.task_type)
        training_result = TrainingResult(
            metrics=dict(result.metrics),
            artifact_paths=dict(result.artifact_paths),
            bundle_dir=run_dir / 'published-model',
            holdout_path=holdout_path,
            holdout_truncated=holdout_truncated,
            _summaries=_unlogged((
                *(() if result.test_counts is None else evaluation_summaries(
                    task_type, result.test_counts, threshold,
                    class_names(result.best_trainer.target_encoder))),
                *at_k_summaries(result.metrics)), result.metrics),
        )
        holdout_counts = None if result.published_counts is None else (
            baseline(result.published_counts).to_dict())
        del result
        _collect_garbage('fit')
        return self._finish_fit(
            training_result,
            predictive_query=train_table.query,
            snapshot_id=snapshot.id,
            graph_config=snapshot.graph_config,
            training_table_generation=train_table.generation_plan,
            export_config=export_config,
            scoring_defaults=scoring_defaults,
            model_plan=effective_plan,
            provenance=dict(
                train_table=train_table,
                warm_start=None if warm_start is None else parent_model(
                    warm_start._training_result.bundle_dir,
                    warm_start_bundle_id),
                started_at=started_at, fit_mode=fit_mode,
                tracking=tracking.pointer(),
                task_type=task_type,
                holdout_path=holdout_path,
                holdout_truncated=holdout_truncated,
                holdout_counts=holdout_counts),
            deps=deps,
        )

    def _finish_fit(
        self,
        training_result: TrainingResult,
        *,
        predictive_query: str,
        snapshot_id: str,
        training_table_generation: TrainingTableGenerationPlan,
        export_config: Optional[ModelOutputConfig],
        graph_config: Optional[Dict[str, Any]] = None,
        model_plan: Optional[ModelPlan] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        deps: Optional[Dependencies] = None,
        scoring_defaults: Optional[ScoringDefaults] = None,
    ) -> TrainingResult:
        """Commit durable fit outputs before making this trainer usable."""
        from kumo_core.artifacts import bundle_id_from_manifest

        from kumo.engine.trainer._provenance import write_provenance

        effective_plan = model_plan if model_plan is not None else self._model_plan
        write_training_spec(
            training_result.bundle_dir,
            predictive_query=predictive_query,
            model_plan=effective_plan,
            snapshot_id=snapshot_id,
            training_table_generation=training_table_generation,
            graph_config=graph_config,
        )
        if scoring_defaults is not None:
            write_scoring_defaults(training_result.bundle_dir,
                                   scoring_defaults)
            training_result = replace(training_result,
                                      scoring_defaults=scoring_defaults)
        training_result = _bundle_holdout(training_result)
        try:
            training_result = replace(
                training_result,
                bundle_id=bundle_id_from_manifest(training_result.bundle_dir))
        except Exception as error:
            warnings.warn(f"Could not compute the bundle id: {error}",
                          UserWarning, stacklevel=3)
        if provenance is not None and training_result.bundle_id is not None:
            try:
                write_provenance(
                    training_result.bundle_dir,
                    bundle_id=training_result.bundle_id,
                    dependencies=self._dependencies(training_result, deps
                                                    or Dependencies()),
                    **provenance)
            except Exception as error:
                warnings.warn(
                    f"Publishing the model bundle without _provenance.json: "
                    f"{error}", UserWarning, stacklevel=3)
        if export_config is not None:
            training_result = replace(
                training_result, published_model=_publish(
                    training_result.bundle_dir, training_result.bundle_id,
                    training_result.metrics, export_config))

        training_result = replace(training_result, _trainer=self)
        self._training_result = training_result
        self._snapshot_id = snapshot_id
        self._graph_config = graph_config
        self._training_query = predictive_query
        self._training_table_generation_plan = training_table_generation

        return training_result

    def _dependencies(self, result: TrainingResult,
                      deps: Dependencies) -> List[Dependency]:
        from kumo.engine._version import __version__

        deps.add(dependency.predictor(TRAINER_PREDICTOR))
        text = require_text_encoders(deps, self._embedding_models,
                                     _encoders(result.bundle_dir))
        deps.add(
            dependency.package('kumo-engine', __version__,
                               ['text'] if text else []))
        return deps.records

    @_without_torch_scatter_hint
    def predict(
        self,
        source: Union[GraphSnapshot, PredictionTable, None] = None,
        /,
        *,
        snapshot: Optional[GraphSnapshot] = None,
        entities: Optional[SourceTable] = None,
        prediction_time: Optional[datetime] = None,
        output_config: Optional[PredictionOutputConfig] = None,
        num_classes_to_return: Optional[int] = None,
        num_workers: int = 1,
        binary_classification_threshold: Optional[float] = None,
        prediction_table: Optional[PredictionTable] = None,
        mlops: Optional[Any] = None,
    ) -> PredictionResult:
        r"""Score a snapshot and return durable file-backed results.

        ``predict(snapshot)`` scores the trained query's population at
        ``prediction_time`` (as :meth:`PredictiveQuery.generate_prediction_table`);
        ``entities`` is a source table of ``ENTITY1`` (and ``TIMESTAMP``
        where the task needs one) rows to score instead. It warns when the
        snapshot was taken of another graph than the one the model was
        trained on. ``predict(prediction_table, snapshot=...)`` scores a
        prediction table generated beforehand, as before.

        The trainer must contain fitted state from :meth:`fit`, :meth:`load`,
        or :meth:`load_from_connector`. Without ``output_config``,
        predictions remain on local disk.

        ``num_workers`` controls scorer processes, not DataLoader workers.
        The default stays in-process; values above one spawn independent
        scorers over disjoint partitions, capped by the prediction population.
        CUDA execution requires one visible device per concurrent scorer.

        ``num_classes_to_return`` must be a positive integer when supplied.
        Link predictions require an explicit count of at most 500. Embeddings
        alone do not require a count.

        Binary classification always returns ``True_PROB``; ``TARGET_PRED``
        is set only with a threshold, else null. Multilabel classification
        defaults to 0.5. Both accept thresholds from 0 to 1; thresholds
        change predicted labels without changing class scores.

        Omitted arguments fall back to the bundle's :attr:`scoring_defaults`.

        ``result.manifest`` records the run (``outputs`` lists where each
        output was written). With MLOps on (``mlops`` or a
        `kumo.mlops` block), the outputs and manifest are also stored under
        ``<artifact root>/predictions/<id>/``.
        """
        from kumo_core.prediction_options import (
            validate_num_classes_to_return,
            validate_prediction_workers,
        )

        from kumo.engine._mlops import resolve

        settings = resolve(mlops)
        validate_prediction_workers(num_workers)
        validate_num_classes_to_return(num_classes_to_return)
        if source is not None and not isinstance(source, GraphSnapshot):
            if prediction_table is not None:
                raise TypeError('predict() got two prediction tables')
            source, prediction_table = None, source
        if source is not None and snapshot is not None:
            raise TypeError('predict() got two snapshots')
        snapshot = snapshot if snapshot is not None else source
        if snapshot is None:
            raise TypeError('predict() needs a snapshot')
        if prediction_table is not None and (entities is not None
                                             or prediction_time is not None):
            raise ValueError('A prediction table already holds its entities '
                             'and times; pass it or entities=/'
                             'prediction_time=, not both.')
        table = (prediction_table if prediction_table is not None else
                 self._prediction_table(
                     snapshot, entities, prediction_time, strict=(
                         settings is not None
                         and settings.on_error == 'raise')))
        with system_metrics(None if settings is None else settings.tracking):
            return self._predict(
                table, snapshot=snapshot, output_config=output_config,
                num_classes_to_return=num_classes_to_return,
                num_workers=num_workers,
                binary_classification_threshold=(
                    binary_classification_threshold),
                settings=settings)

    def _prediction_table(self, snapshot: GraphSnapshot,
                          entities: Optional[SourceTable],
                          prediction_time: Optional[datetime],
                          strict: bool = False) -> PredictionTable:
        from kumo.engine.pquery.predictive_query import PredictiveQuery

        if self._training_query is None:
            raise RuntimeError(
                "Trainer.predict() requires a prior Trainer.fit() or "
                "Trainer.load() call on this same Trainer instance.")
        if entities is not None and prediction_time is not None:
            raise ValueError("Pass prediction_time or entities, not both; "
                             "put anchor times in the entities' TIMESTAMP.")
        trained = (self._snapshot_id or '').partition('@')[0]
        if trained and snapshot._graph_key != trained:
            report_other_graph(
                f"Snapshot {snapshot.id} was taken of graph "
                f"{snapshot._graph_key}, but this model was trained on a "
                f"snapshot of graph {trained}.", self._graph_config,
                snapshot.graph_config, stacklevel=3, strict=strict)
        query = PredictiveQuery._from_snapshot(snapshot, self._training_query)
        plan = self._training_table_generation_plan
        return (query.generate_prediction_table(
            snapshot, prediction_time=prediction_time, plan=plan)
                if entities is None else PredictionTable.from_source_table(
                    entities, pq=query, snapshot=snapshot, plan=plan))

    def _predict(
        self,
        prediction_table: PredictionTable,
        *,
        snapshot: GraphSnapshot,
        output_config: Optional[PredictionOutputConfig] = None,
        num_classes_to_return: Optional[int] = None,
        num_workers: int = 1,
        binary_classification_threshold: Optional[float] = None,
        settings: Optional[Any] = None,
        extra_fields: Sequence[str] = (),
    ) -> PredictionResult:
        training_result = self._fitted('predict')
        arguments = binary_classification_threshold, num_classes_to_return
        if binary_classification_threshold is None:
            binary_classification_threshold = (
                training_result.scoring_defaults.binary_classification_threshold)
        from kumo_core.pquery import normalize_graph, parse_and_validate_query
        from kumo_core.prediction_options import (
            prediction_query_contract,
            resolve_classification_threshold,
            resolve_num_classes_to_return,
        )
        from kumo_core.types import TaskType

        graph_config = snapshot.graph_config
        contract = prediction_query_contract(prediction_table.query,
                                             graph_config)
        if self._training_query is None or prediction_query_contract(
                self._training_query, graph_config) != contract:
            raise ValueError(
                'Prediction query target/entity contract differs from the trained model'
            )
        task_type = TaskType(contract[1])
        threshold = (None if binary_classification_threshold is None
                     and task_type is TaskType.BINARY_CLASSIFICATION else
                     resolve_classification_threshold(
                         task_type, binary_classification_threshold))
        destinations = (output_config.destinations()
                        if output_config is not None else {})
        artifacts = frozenset(
            PredictionArtifactType(name.upper())
            for name in destinations) or frozenset(
                {PredictionArtifactType.PREDICTIONS})
        artifact_type = (PredictionArtifactType.PREDICTIONS
                         if PredictionArtifactType.PREDICTIONS in artifacts else
                         PredictionArtifactType.EMBEDDINGS)
        # Parse against the selected graph so its topology, feature schema,
        # and index coordinates drive task materialization. The generated
        # population and anchors still come from the immutable PredictionTable.
        native_graph = normalize_graph(graph_config)
        parsed = parse_and_validate_query(prediction_table.query, native_graph)
        if num_classes_to_return is None and parsed.query.top_k is None:
            num_classes_to_return = training_result.scoring_defaults.num_classes_to_return
        requested_classes = num_classes_to_return
        num_classes_to_return = resolve_num_classes_to_return(
            task_type,
            num_classes_to_return,
            top_k=parsed.query.top_k,
            has_predictions=PredictionArtifactType.PREDICTIONS in artifacts,
        )

        run_id = f'{self._id}-predict-{new_run_id()}'
        if output_config is not None:
            for destination in output_config._targets(run_id).values():
                preflight = getattr(output_config.writer(destination),
                                    'preflight_write_files', None)
                if callable(preflight):
                    preflight(output_config.connector, destination,
                              write_mode=output_config.write_mode)

        from kumo_core.encoders.config import EncodersConfig
        from kumo_core.io.json import load_json

        # Loaded once: the embedding check and the in-process scorer share it.
        encoders_path = training_result.artifact_paths.get('encoders')
        encoders = (load_json(encoders_path, EncodersConfig) if encoders_path
                    and Path(encoders_path).is_file() else None)
        require_trained_embedding_models(self._embedding_models, encoders)

        from kumo_core.prediction import (
            GeneratedPredictionTable,
            PredictionTableRequest,
            forecast_task_options,
            materialize_prediction_task,
            score_prediction,
        )

        from kumo.engine.trainer._provenance import count_edges

        started_at = datetime.now(timezone.utc)
        run_dir = self.scratch_dir / 'runs' / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if self._training_table_generation_plan is not None:
            parsed = resolve_query_forecast(
                parsed, native_graph, self._training_table_generation_plan)
        ingest_tt_id = f'{prediction_table.tt_id}-{run_id}'
        compiler_options = prediction_table.compiler_options or {}
        lag_offsets = compiler_options.get('lag_offsets_secs')
        lag_timesteps = (len(lag_offsets) if isinstance(lag_offsets,
                                                        (tuple,
                                                         list)) else None)
        with no_encoder_inference_warnings():
            task_dataset = build_generated_training_task_dataset(
                query=parsed,
                graph=native_graph,
                graph_index_path=str(snapshot.index_path),
                run_id=run_id,
                artifact_dir=str(run_dir),
                tt_id=ingest_tt_id,
                required_stages=prediction_table.required_stages,
                lag_timesteps=lag_timesteps,
                year_over_year='yoy_offsets_secs' in compiler_options,
            )
        request = PredictionTableRequest(
            query=parsed,
            graph=native_graph,
            timestamps=prediction_table.timestamps,
            index_path=str(snapshot.index_path),
            tt_id=ingest_tt_id,
            task_dataset=task_dataset,
            compiler_options=prediction_table.compiler_options,
            forecast_timeframe_step=prediction_table.forecast_timeframe_step,
        )
        generated = GeneratedPredictionTable(
            staging_dir=str(prediction_table.staging_dir),
            table_path=str(prediction_table.table_path),
            row_count=prediction_table.row_count,
            tt_id=ingest_tt_id,
        )
        materialized = materialize_prediction_task(request, generated,
                                                   graph_index_id=snapshot.id)
        timestamps = request.timestamps
        forecast_options = forecast_task_options(request)
        # Scoring reloads the materialized feature schema; drop this copy
        # first so the parent and its forked loader workers never hold both.
        del request, task_dataset

        model_dir = str(
            _stage_model_bundle(
                training_result.artifact_paths,
                run_dir / 'model_bundle',
            ))

        extra_fields = [
            *(f.value for f in (output_config.extra_fields
                                if output_config is not None else [])),
            *extra_fields
        ]
        source_identity = [
            snapshot.id, prediction_table.query, timestamps,
            sorted(kind.value for kind in artifacts), num_classes_to_return,
            threshold, extra_fields
        ]
        selection = score_prediction(
            materialized=materialized,
            model_dir=model_dir,
            run_id=run_id,
            scratch_dir=str(self.scratch_dir),
            artifacts=artifacts,
            top_k=num_classes_to_return,
            num_workers=num_workers,
            extra_fields=({
                kind.value: extra_fields
                for kind in artifacts
            } if extra_fields else None),
            started=datetime.now(timezone.utc),
            graph_snapshot_id=snapshot.id,
            source_config_sha256=hashlib.sha256(
                json.dumps(source_identity, sort_keys=True,
                           default=str).encode()).hexdigest(),
            binary_threshold=threshold,
            forecast_options=forecast_options,
            embedding_models=self._embedding_models,
            encoders=encoders,
            accelerator=self._accelerator,
            count_edges=(None if settings is None else count_edges(
                training_result.manifest, task_type)),
        )

        del encoders
        _collect_garbage('predict')
        result = PredictionResult(selection, artifact_type)
        manifest = _run_manifest.start(
            result=result, trainer=self, snapshot=snapshot,
            prediction_table=prediction_table,
            task_type=task_type, scoring=RunScoring(
                binary_classification_threshold=threshold,
                threshold_source=_option_source(
                    arguments[0], binary_classification_threshold, threshold,
                    'default'),
                num_classes_to_return=num_classes_to_return,
                count_source=_option_source(arguments[1], requested_classes,
                                            num_classes_to_return, 'query'),
                num_workers=num_workers,
            ), started_at=started_at.isoformat(timespec='seconds'))
        _run_manifest.finish(result, manifest, output_config, settings,
                             training_result, snapshot)
        return result

    def evaluate(self, run: PredictionResult, *,
                 labels_snapshot: GraphSnapshot,
                 mlops: Optional[Any] = None) -> 'EvaluationResult':
        r"""Score a run against labels regenerated on ``labels_snapshot``, a
        snapshot newer than the one the run scored, holding its outcomes.

        With MLOps tracking on (``mlops`` or a `kumo.mlops` block), the
        result is logged as a new monitoring run of ``run``.
        """
        from kumo.engine._mlops import resolve
        from kumo.engine._monitoring import evaluate, track

        settings = resolve(mlops)
        with system_metrics(None if settings is None else settings.tracking):
            result = evaluate(self, run, labels_snapshot)
        track(result, run, settings, self.result)
        return result

    def score_drift(
        self,
        run: PredictionResult,
        *,
        previous: Optional[PredictionResult] = None,
        snapshot: Optional[GraphSnapshot] = None,
        training_snapshot: Optional[GraphSnapshot] = None,
        mlops: Optional[Any] = None,
    ) -> 'DriftReport':
        r"""Compare a run's scores with the holdout and a previous run.

        Args:
            previous: An earlier run, for run-over-run statistics.
            snapshot: The snapshot the run scored. By default, the one the
                run's manifest names, from the local snapshot store or,
                with MLOps on, the artifact root.
            training_snapshot: The snapshot this model trained on; found
                like ``snapshot`` by default.
            mlops: Logs the report as a new monitoring run of ``run`` when
                tracking is on, like `evaluate`.
        """
        from kumo.engine._mlops import resolve
        from kumo.engine._monitoring import score_drift, track

        settings = resolve(mlops)
        with system_metrics(None if settings is None else settings.tracking):
            report = score_drift(self, run, previous=previous,
                                 snapshot=snapshot,
                                 training_snapshot=training_snapshot,
                                 root=None if settings is None else
                                 settings.root)
        track(report, run, settings, self.result)
        return report

    @classmethod
    def load(
        cls,
        bundle_path: str | Path,
        embedding_models: Optional[EmbeddingModelStore] = None,
        *,
        accelerator: Literal['cpu', 'gpu', 'mps', 'auto'] = 'auto',
    ) -> Self:
        r"""Restore a trainer from an existing local model bundle.

        Its text columns need the same word-embedding models it was trained
        with, registered in ``embedding_models`` under any name or installed.
        ``accelerator`` selects the device for this instance, defaulting to
        ``auto`` independently of the device used to train the model.
        """
        return cls._load_bundle(Path(bundle_path), embedding_models,
                                accelerator=accelerator)

    @classmethod
    def load_from_connector(
        cls,
        model_registry: Connector,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        embedding_models: Optional[EmbeddingModelStore] = None,
        accelerator: Literal['cpu', 'gpu', 'mps', 'auto'] = 'auto',
    ) -> Self:
        r"""Fetch a trusted publication, then restore its training state.

        Supply exactly one of ``path`` or ``location``, using the address
        used for publication. Path addresses are interpreted by the connector.
        The bundle must have been exported by a local SDK that saves trainer
        state. Legacy bundles require a new fit/export; queries are not guessed.
        Prediction requires a compatible query and an explicit graph snapshot;
        the bundle does not restore graph storage or source data.
        Hashes detect corruption, not publisher authentication. Load only models
        from trusted publishers; checkpoint deserialization happens at predict.
        ``accelerator`` selects the device for this instance (default: ``auto``).
        A `kumo.MLOps` ``model_registry`` reads ``path`` under its artifact
        root (``models/<id>``) and ``location`` from the registry its
        tracker registers in.
        """
        from kumo.engine._mlops import MLOps, on_root

        if isinstance(model_registry, MLOps) and location is not None:
            load = getattr(model_registry.tracking, 'load_model', None)
            if path is not None or load is None:
                raise TypeError(
                    'Loading a registered model from a kumo.MLOps takes '
                    'location= and its tracking=MlflowTracking(...) or '
                    'SnowflakeTracking(connector, ...).')
            return load(location, embedding_models=embedding_models,
                        accelerator=accelerator)
        if path is not None:
            model_registry, path = on_root(model_registry, path)
        address = ModelOutputConfig(model_registry, path=path,
                                    location=location)
        bundle_dir = trainer_scratch_dir(new_run_id()) / 'fetched-model'
        sink = model_sink(model_registry, address.path)
        if not isinstance(sink, ModelFetchingSink):
            raise TypeError(
                'model_registry.sink_backend must implement ModelFetchingSink')
        alias = None if location is None else location.alias
        if alias is not None and not isinstance(sink, ModelAliasingSink):
            raise NotImplementedError(
                f'{type(model_registry).__name__} cannot fetch models by '
                f'alias')
        published = sink.fetch_model(cast(SourceConnector, model_registry),
                                     str(bundle_dir), path=address.path,
                                     location=address.location)
        trainer = cls._load_bundle(bundle_dir, embedding_models,
                                   accelerator=accelerator)
        warn_if_other_bundle(getattr(published, 'metadata', None), trainer)
        if location is not None:
            trainer._from_registry(published, alias)
        return trainer

    def _from_registry(self, model: PublishedModel,
                       alias: Optional[str]) -> None:
        self._training_result = replace(self._fitted('predict'),
                                        registered_model=model,
                                        registered_alias=alias)

    @classmethod
    def _load_bundle(
        cls,
        bundle_dir: Path,
        embedding_models: Optional[EmbeddingModelStore] = None,
        *,
        accelerator: Literal['cpu', 'gpu', 'mps', 'auto'] = 'auto',
    ) -> Self:
        bundle_dir = bundle_dir.resolve()
        paths, bundle_id = read_bundle_artifact_paths(bundle_dir)
        spec = read_training_spec(bundle_dir)
        from kumo_core.io.json import from_json_dict
        plan = from_json_dict(spec.model_plan, ModelPlan)
        metrics_path = bundle_dir / 'metrics.json'
        metrics = (json.loads(metrics_path.read_text(
            encoding='utf-8')) if metrics_path.is_file() else {})
        trainer = cls(plan, embedding_models, accelerator=accelerator)
        trainer._snapshot_id = spec.snapshot_id
        trainer._graph_config = spec.graph_config
        trainer._training_query = spec.predictive_query
        trainer._training_table_generation_plan = from_json_dict(
            spec.training_table_generation, TrainingTableGenerationPlan)
        from kumo.engine._mlops import resolve
        from kumo.engine.trainer._provenance import read_provenance

        trainer._embedding_models = with_text_encoders(
            trainer._embedding_models, _encoders(bundle_dir),
            Dependencies.of(read_provenance(bundle_dir), resolve()))
        holdout_path, holdout_truncated = _listed_holdout(bundle_dir)
        trainer._training_result = TrainingResult(
            metrics, paths, bundle_dir, holdout_path=holdout_path,
            holdout_truncated=holdout_truncated, bundle_id=bundle_id,
            scoring_defaults=read_scoring_defaults(bundle_dir, paths),
            _trainer=trainer)
        return trainer


def _encoders(bundle_dir: Path) -> Any:
    from kumo_core.encoders.config import EncodersConfig
    from kumo_core.io.json import load_json

    path = listed_file(bundle_dir, 'encoders')
    return (load_json(str(path), EncodersConfig)
            if path is not None and path.is_file() else None)


def _set_alias(connector: Connector, location: ModelRegistryLocation,
               alias: str) -> None:
    sink = connector.sink_backend
    if not isinstance(sink, ModelAliasingSink):
        raise NotImplementedError(
            f'{type(connector).__name__} cannot set model aliases')
    sink.set_model_alias(cast(SourceConnector, connector), location, alias)


def warn_if_other_bundle(metadata: Optional[Mapping[str, Any]],
                         trainer: Trainer) -> None:
    recorded = (metadata or {}).get(BUNDLE_ID_KEY)
    current = None if trainer.result is None else trainer.result.id
    if recorded and current and recorded != current:
        warnings.warn(
            f"The registry records bundle {recorded} for this version, but "
            f"the fetched bundle is {current}.", UserWarning,
            stacklevel=3)


def registry_pointers(bundle_id: Optional[str]) -> Dict[str, str]:
    return {BUNDLE_ID_KEY: bundle_id} if bundle_id else {}
