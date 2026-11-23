# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded explanation replay over an already scored prediction population."""

import posixpath
from typing import Mapping

from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.encoders.config import EncodersConfig
from kumo_core.explain.base import Explainer
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.explain.subgraph_collector import SubgraphCollector
from kumo_core.io import filesystem
from kumo_core.io.json import load_json
from kumo_core.prediction_explanation_result import (
    PredictionExplanationCoverage,
    PredictionExplanationOptions,
    PredictionExplanationResult,
    PredictionExplanationSupport,
    ProducerOutcome,
)
from kumo_core.task_dataset import (
    DiskgraphTaskManifest,
    TaskDatasetMaterializeResult,
)
from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
from kumo_core.training.config import (
    FitMode,
    LinkPredModuleType,
    TrainerConfig,
)
from kumo_core.training.runtime import LocalExplainRuntime
from kumo_core.types.enums import ExplainMethod, TaskType
from kumoapi.explain.column_analysis import ColumnAnalysisOutput
from kumoapi.explain.common import EntityMappings
from kumoapi.train import TrainingStage


def prediction_explanation_support(
    task_type: TaskType,
    module: LinkPredModuleType | None = None,
) -> PredictionExplanationSupport:
    """Query eligibility without generation; omit module for task eligibility."""
    return PredictionExplanationSupport(
        task_type in ColumnAnalysis.supported_task_types,
        SubgraphCollector.supports(task_type, module))


def prediction_model_support(
    task_type: TaskType,
    model_dir: str,
) -> PredictionExplanationSupport:
    """Resolve eligibility from the published model before loading its weights."""
    if not task_type.is_link_pred:
        return prediction_explanation_support(task_type)
    config = load_json(posixpath.join(model_dir, 'config.json'), TrainerConfig)
    return prediction_explanation_support(
        task_type, config.module_kwargs.module
        if config.module_kwargs is not None else None)


def _cohort_error(output_dir: str) -> str | None:
    path = posixpath.join(output_dir, 'column_analysis', 'pred',
                          'column_analysis.json')
    try:
        if not filesystem.exists(path):
            return f'Missing cohort artifact: {path}'
        load_json(path, ColumnAnalysisOutput)
    except Exception as error:
        return f'Invalid cohort artifact {path}: {type(error).__name__}: {error}'
    return None


def _sample_inventory(output_dir: str) -> tuple[int | None, str | None]:
    """Check saved graph membership once and count retained mapped graphs."""
    collector = posixpath.join(output_dir, 'subgraph_collector', 'pred')
    mapping_path = posixpath.join(collector, 'strategy_mappings.json')
    try:
        if not filesystem.exists(mapping_path):
            return None, f'Missing prediction selection artifact: {mapping_path}'
        mappings = load_json(mapping_path, EntityMappings)
        graphs = posixpath.join(collector, 'subgraph_json')
        ids = {artifact for group in mappings.mappings.values()
               for artifact in group}
        retained = sum(filesystem.exists(posixpath.join(graphs, f'{item}.json'))
                       for item in ids)
        if not ids:
            return 0, f'Empty prediction selection artifact: {mapping_path}'
        if retained != len(ids):
            return retained, (f'{len(ids) - retained} of {len(ids)} sampled graphs '
                              f'missing under {graphs}')
    except Exception as error:
        return None, f'Invalid prediction selection {mapping_path}: {error}'
    return retained, None


def _outcome(supported: bool, artifact_error: str | None,
             producer: Explainer | None, replay_error: str | None,
             ) -> ProducerOutcome:
    if not supported:
        return ProducerOutcome('unsupported')
    if producer is not None and producer.failure is not None:
        return ProducerOutcome('failed', producer.failure)
    if artifact_error is not None:
        error = (f'{replay_error}; {artifact_error}' if replay_error
                 else artifact_error)
        return ProducerOutcome('failed', error)
    if producer is None or not producer.completed:
        return ProducerOutcome('failed', replay_error or
                               'Explanation producer did not complete')
    return ProducerOutcome('complete')


def explain_prediction(
    *,
    materialized: TaskDatasetMaterializeResult,
    model_dir: str,
    run_id: str,
    output_dir: str,
    top_k: int | None = None,
    binary_threshold: float | None = None,
    task_options: Mapping[str, int | None] | None = None,
    embedding_models: EmbeddingModelStore | None = None,
    encoders: EncodersConfig | None = None,
    accelerator: str = 'auto',
    options: PredictionExplanationOptions | None = None,
) -> PredictionExplanationResult:
    """Generate cohorts and sampled graphs with bounded prediction replay.

    Call after scoring succeeds. ``output_dir`` is the run's ``xai`` directory;
    producer outputs use the ``pred`` stage. A fresh, unsharded trainer reads
    exactly the prediction task bundle and published checkpoint used to score.
    Writes no prediction outputs. Cohort and graph row budgets are independent.
    ``model_dir`` must be a per-run copy, because collector staging uses it.
    Return independent producer outcomes, retaining successful artifacts.
    """
    manifest = DiskgraphTaskManifest.load(materialized.manifest_path)
    task_type = TaskType(manifest.task_type)
    support = prediction_explanation_support(task_type)
    options = options or PredictionExplanationOptions()
    if not support.cohorts and not support.samples:
        return PredictionExplanationResult(ProducerOutcome('unsupported'),
                                           ProducerOutcome('unsupported'))
    producers: tuple[Explainer, ...] = ()
    replay_error: str | None = None
    trainer: DiskgraphNestedTrainer | None = None
    try:
        if task_type.is_link_pred:
            support = prediction_model_support(task_type, model_dir)
            if not support.samples:
                return PredictionExplanationResult(ProducerOutcome('unsupported'),
                                                   ProducerOutcome('unsupported'))
        trainer = DiskgraphNestedTrainer.load_prediction(
            manifest=manifest,
            model_dir=model_dir,
            bundle_dir=materialized.bundle_dir,
            run_id=run_id,
            encoders=encoders,
            task_options=task_options,
            binary_threshold=binary_threshold,
            explain_runtime=LocalExplainRuntime(),
            embedding_models=embedding_models,
            accelerator=accelerator,
        )
        methods: list[ExplainMethod] = []
        if support.cohorts:
            methods.append(ExplainMethod.COLUMN_ANALYSIS)
        if support.samples:
            methods.append(ExplainMethod.SUBGRAPH_COLLECTOR)
        # This published directory already is the selected fit/refit; do not
        # select a sibling refit from its historical training configuration.
        trainer.predict_setup(fit_mode_to_use=FitMode.fit, task_manifest=manifest)
        producers = trainer.explain(
            methods,
            stage=TrainingStage.PRED,
            fit_mode_to_use=FitMode.fit,
            root_dir=output_dir,
            top_k=top_k,
            collector_formats=['nested_json', 'output'],
            options=options,
        )
    except Exception as error:
        replay_error = f'{type(error).__name__}: {error}'
        if trainer is not None:
            producers = trainer.explanation_producers
    cohort_producer = next((producer for producer in producers
                            if isinstance(producer, ColumnAnalysis)), None)
    sample_producer = next((producer for producer in producers
                            if isinstance(producer, SubgraphCollector)), None)
    checkpoint: str | None = None
    if trainer is not None:
        try:
            checkpoint = filesystem.local_path(trainer.best_ckpt).relative_to(
                filesystem.local_path(model_dir)).as_posix()
        except Exception:
            # Failed initialization may not resolve a checkpoint at all.
            pass
    sampled_graphs, sample_error = (_sample_inventory(output_dir)
                                    if support.samples else (None, None))
    return PredictionExplanationResult(
        _outcome(support.cohorts,
                 _cohort_error(output_dir) if support.cohorts else None,
                 cohort_producer, replay_error),
        _outcome(support.samples, sample_error, sample_producer, replay_error),
        PredictionExplanationCoverage(
            cohort_rows_processed=(cohort_producer.processed_rows
                                   if cohort_producer is not None else None),
            sampled_graphs=sampled_graphs,
            checkpoint_path=checkpoint))
