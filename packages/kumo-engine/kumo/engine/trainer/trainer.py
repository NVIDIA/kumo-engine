# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Synchronous local model training, prediction, and bundle loading."""

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional, cast

import pandas as pd
from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.model_plan import ModelPlan as CoreModelPlan
from kumo_core.model_plan import inferred_or_missing_type_to_none
from kumo_core.model_plan_conversion import public_model_plan_from_internal
from kumo_core.pquery import (
    build_generated_training_task_dataset,
    resolve_query_forecast,
)
from kumo_core.train.artifact_manifest import PredictionSelectionRef
from kumo_core.training.explain_aggregation import aggregate_subgraph_features
from kumo_core.training.workflow import TrainingWorkflowRequest
from kumo_core.types import ExplainMethod, PredictionArtifactType
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.plugin import (
    ModelFetchingSink,
    ModelRegistryLocation,
    PublishedModel,
    SourceConnector,
)
from kumoapi.train import TrainingStage
from typing_extensions import Self

from kumo.engine._embeddings import (
    no_encoder_inference_warnings,
    require_embedding_models,
    require_trained_embedding_models,
)
from kumo.engine._scratch import new_run_id, trainer_scratch_dir
from kumo.engine.connector.base import Connector
from kumo.engine.graph import GraphSnapshot
from kumo.engine.pquery.prediction_table import PredictionTable
from kumo.engine.pquery.training_table import TrainingTable
from kumo.engine.trainer._spec import (
    read_bundle_artifact_paths,
    read_training_spec,
    write_training_spec,
)
from kumo.engine.trainer.output_config import (
    ModelOutputConfig,
    PredictionOutputConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    r"""Metrics and on-disk artifacts produced by :meth:`Trainer.fit`.

    `published_model` records the destination's assigned name, version and URI
    when `fit()` exports the bundle. It is `None` without an export.

    `holdout_path` is the holdout Parquet file written by `fit()`, or `None`
    when the task produced none (also for results of `Trainer.load`). It is
    not part of the published model bundle. The file is capped at 1 GiB in
    memory; `holdout_truncated` is true when rows were dropped to fit, and
    `None` when the size record could not be read.
    """

    metrics: Dict[str, float]
    artifact_paths: Dict[str, str]
    bundle_dir: Path
    published_model: Optional[PublishedModel] = None
    holdout_path: Optional[Path] = None
    holdout_truncated: Optional[bool] = False

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
    """

    selection: PredictionSelectionRef
    artifact_type: PredictionArtifactType

    @property
    def num_rows(self) -> int:
        from kumo_core.train import artifact_manifest
        selected = artifact_manifest.load_prediction_selection(
            self.selection, validate_objects=False)
        return next(item.num_rows for item in selected.artifacts
                    if item.artifact_type is self.artifact_type)

    @property
    def files(self) -> tuple[str, ...]:
        from kumo_core.train import artifact_manifest
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
        self._export(output_config)

    def _export(self, output_config: PredictionOutputConfig) -> None:
        write_files = getattr(output_config.connector.sink_backend,
                              'write_files', None)
        if not callable(write_files):
            raise NotImplementedError(
                'Prediction export requires FileArtifactSink.write_files')
        outputs = []
        for name, destination in output_config.destinations().items():
            kind = PredictionArtifactType(name.upper())
            files = PredictionResult(self.selection, kind).files
            outputs.append((kind, destination, files))
        for kind, destination, files in outputs:
            write_files(output_config.connector, destination, list(files),
                        output_config.write_mode,
                        export_id=f'{self.selection.job_id}-{kind.value}')


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


def _stage_model_bundle(artifacts: Mapping[str, str], dest: Path) -> Path:
    from kumo_core.artifacts import stage_model_bundle
    return stage_model_bundle(artifacts, dest)


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

    def fit(
        self,
        train_table: TrainingTable,
        *,
        snapshot: GraphSnapshot,
        export_config: Optional[ModelOutputConfig] = None,
        warm_start: Optional[Self] = None,
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
            warm_start: a fitted trainer or one restored with ``load`` or
                ``load_from_connector``. Reuses its selected weights, optimizer
                state, fitted encoders, architecture and neighbor sampling;
                this trainer's training controls still apply. Task types must
                match. Forecasting warm starts are unsupported. Older bundles
                may contain an AutoML search space instead of the fitted
                architecture; they support prediction but are unreliable
                warm-start sources.
        """
        from kumo_core.automl.runtime import (
            inherit_warm_start_model_plan,
            validate_warm_start_task_type,
        )
        from kumo_core.io.json import load_json
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

        from kumo.engine.trainer._runtime import local_training_bindings

        if export_config is not None and not callable(
                getattr(export_config.connector.sink_backend, 'publish_model',
                        None)):
            raise NotImplementedError(
                f'{type(export_config.connector).__name__} cannot publish '
                f'models')
        run_id = f'{self._id}-fit-{new_run_id()}'
        run_dir = self.scratch_dir / 'runs' / run_id

        model_plan = _to_core_model_plan(self._model_plan)
        warm_start_artifacts = None
        effective_plan = self._model_plan
        if warm_start is not None:
            if warm_start._training_result is None:
                raise ValueError(
                    "warm_start requires a fitted or loaded Trainer")
            warm_start_artifacts = read_bundle_artifact_paths(
                warm_start._training_result.bundle_dir)
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
        if warm_start_artifacts is not None:
            source_schema = json.loads(
                Path(warm_start_artifacts['prediction_schema']).read_text())
            validate_warm_start_task_type(
                TaskType(workflow.query.task_type),
                TaskType(source_schema['task_type']),
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
        task_dataset = build_generated_training_task_dataset(
            query=workflow.query,
            graph=workflow.graph,
            graph_index_path=workflow.prepared_graph.index_path,
            run_id=run_id,
            artifact_dir=str(run_dir),
            tt_id=ingest_tt_id,
            required_stages=train_table.required_stages,
            lag_timesteps=(None
                           if lag_timesteps is None else int(lag_timesteps)),
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
        training_input = ResolvedTrainingInput(
            task=materialized.task,
            model_plan=materialized.model_plan,
            home_dir=str(run_dir),
            task_dataset_bundle_dir=materialized.bundle_dir,
            warm_start_artifacts=warm_start_artifacts,
        )
        result = run_callable_training(
            training_input,
            bindings=local_training_bindings(
                training_input, run_name=run_id,
                embedding_models=self._embedding_models,
                accelerator=self._accelerator),
            options=CallableTrainingOptions(),
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

        holdout_path, holdout_truncated = _find_holdout(
            result.best_trainer.log_dir)
        training_result = TrainingResult(
            metrics=dict(result.metrics),
            artifact_paths=dict(result.artifact_paths),
            bundle_dir=run_dir / 'published-model',
            holdout_path=holdout_path,
            holdout_truncated=holdout_truncated,
        )
        del result
        _collect_garbage('fit')
        return self._finish_fit(
            training_result,
            predictive_query=train_table.query,
            snapshot_id=snapshot.id,
            training_table_generation=train_table.generation_plan,
            export_config=export_config,
            model_plan=effective_plan,
        )

    def _finish_fit(
        self,
        training_result: TrainingResult,
        *,
        predictive_query: str,
        snapshot_id: str,
        training_table_generation: TrainingTableGenerationPlan,
        export_config: Optional[ModelOutputConfig],
        model_plan: Optional[ModelPlan] = None,
    ) -> TrainingResult:
        """Commit durable fit outputs before making this trainer usable."""
        effective_plan = model_plan if model_plan is not None else self._model_plan
        write_training_spec(
            training_result.bundle_dir,
            predictive_query=predictive_query,
            model_plan=effective_plan,
            snapshot_id=snapshot_id,
            training_table_generation=training_table_generation,
        )
        if export_config is not None:
            spec = read_training_spec(training_result.bundle_dir)
            published = export_config.connector.sink_backend.publish_model(
                export_config.connector,
                str(training_result.bundle_dir),
                path=export_config.path,
                location=export_config.location,
                metadata={
                    name: str(getattr(spec, name))
                    for name in ('spec_version', 'predictive_query',
                                 'snapshot_id')
                },
                metrics=training_result.metrics,
            )
            training_result = replace(training_result,
                                      published_model=published)

        self._training_result = training_result
        self._snapshot_id = snapshot_id
        self._training_query = predictive_query
        self._training_table_generation_plan = training_table_generation

        return training_result

    def predict(
        self,
        prediction_table: PredictionTable,
        *,
        snapshot: GraphSnapshot,
        output_config: Optional[PredictionOutputConfig] = None,
        num_classes_to_return: Optional[int] = None,
        num_workers: int = 1,
        binary_classification_threshold: Optional[float] = None,
    ) -> PredictionResult:
        r"""Run batch prediction and return durable file-backed results.

        The trainer must contain fitted state from :meth:`fit`, :meth:`load`,
        or :meth:`load_from_connector`. ``snapshot`` selects graph data and
        features independently of the table's retained population and anchor
        times. Without ``output_config``, predictions remain on local disk.

        ``num_workers`` controls scorer processes, not DataLoader workers.
        The default stays in-process; values above one spawn independent
        scorers over disjoint partitions, capped by the prediction population.
        CUDA execution requires one visible device per concurrent scorer.

        ``num_classes_to_return`` must be a positive integer when supplied.
        Link predictions require an explicit count of at most 500. Embeddings
        alone do not require a count.

        Binary classification requires an explicit threshold. Multilabel
        classification defaults to 0.5. Both accept thresholds from 0 to 1;
        thresholds change predicted labels without changing class scores.
        """
        from kumo_core.prediction_options import (
            validate_num_classes_to_return,
            validate_prediction_workers,
        )

        validate_prediction_workers(num_workers)
        validate_num_classes_to_return(num_classes_to_return)
        if self._training_result is None:
            raise RuntimeError(
                "Trainer.predict() requires a prior Trainer.fit() call on "
                "this same Trainer instance.")
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
        threshold = resolve_classification_threshold(
            task_type, binary_classification_threshold, binary_required=True)
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
        num_classes_to_return = resolve_num_classes_to_return(
            task_type,
            num_classes_to_return,
            top_k=parsed.query.top_k,
            has_predictions=PredictionArtifactType.PREDICTIONS in artifacts,
        )

        write_files = None
        if output_config is not None:
            write_files = getattr(output_config.connector.sink_backend,
                                  'write_files', None)
            if not callable(write_files):
                raise NotImplementedError(
                    'Prediction export requires FileArtifactSink.write_files; '
                    'this sink only supports in-memory table output')
            preflight = getattr(output_config.connector.sink_backend,
                                'preflight_write_files', None)
            if callable(preflight):
                for destination in destinations.values():
                    preflight(output_config.connector, destination,
                              write_mode=output_config.write_mode)

        from kumo_core.encoders.config import EncodersConfig
        from kumo_core.io.json import load_json

        # Loaded once: the embedding check and the in-process scorer share it.
        encoders_path = self._training_result.artifact_paths.get('encoders')
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

        run_id = f'{self._id}-predict-{new_run_id()}'
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
                self._training_result.artifact_paths,
                run_dir / 'model_bundle',
            ))

        extra_fields = ([f.value for f in output_config.extra_fields]
                        if output_config is not None else [])
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
        )

        del encoders
        _collect_garbage('predict')
        result = PredictionResult(selection, artifact_type)
        if output_config is not None:
            result._export(output_config)
        return result

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
        """
        address = ModelOutputConfig(model_registry, path=path,
                                    location=location)
        bundle_dir = trainer_scratch_dir(new_run_id()) / 'fetched-model'
        sink = getattr(model_registry, 'sink_backend')
        if not isinstance(sink, ModelFetchingSink):
            raise TypeError(
                'model_registry.sink_backend must implement ModelFetchingSink')
        sink.fetch_model(cast(SourceConnector, model_registry), str(bundle_dir),
                         path=address.path, location=address.location)
        return cls._load_bundle(bundle_dir, embedding_models,
                                accelerator=accelerator)

    @classmethod
    def _load_bundle(
        cls,
        bundle_dir: Path,
        embedding_models: Optional[EmbeddingModelStore] = None,
        *,
        accelerator: Literal['cpu', 'gpu', 'mps', 'auto'] = 'auto',
    ) -> Self:
        bundle_dir = bundle_dir.resolve()
        paths = read_bundle_artifact_paths(bundle_dir)
        spec = read_training_spec(bundle_dir)
        from kumo_core.io.json import from_json_dict
        plan = from_json_dict(spec.model_plan, ModelPlan)
        metrics_path = bundle_dir / 'metrics.json'
        metrics = (json.loads(metrics_path.read_text(
            encoding='utf-8')) if metrics_path.is_file() else {})
        trainer = cls(plan, embedding_models, accelerator=accelerator)
        trainer._snapshot_id = spec.snapshot_id
        trainer._training_query = spec.predictive_query
        trainer._training_table_generation_plan = from_json_dict(
            spec.training_table_generation, TrainingTableGenerationPlan)
        trainer._training_result = TrainingResult(metrics, paths, bundle_dir)
        return trainer
