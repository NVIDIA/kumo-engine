# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded explanation replay over an already scored prediction population."""

import posixpath
from dataclasses import dataclass
from typing import Mapping

from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.encoders.config import EncodersConfig
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.explain.subgraph_collector import SubgraphCollector
from kumo_core.io import filesystem
from kumo_core.io.json import load_json
from kumo_core.resolved_task import ResolvedNumericalTask
from kumo_core.task_dataset import (
    DiskgraphTaskManifest,
    TaskDatasetMaterializeResult,
)
from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
from kumo_core.train.trainer import load_encoders
from kumo_core.training.config import FitMode, LinkPredModuleType
from kumo_core.training.runtime import (
    LocalDiskgraphTrainingRuntime,
    LocalExplainRuntime,
)
from kumo_core.types.enums import ExplainMethod, TaskType
from kumoapi.explain.common import EntityMappings
from kumoapi.train import TrainingStage


@dataclass(frozen=True)
class PredictionExplanationSupport:
    """Producer eligibility, independently of retained artifact availability."""

    cohorts: bool
    samples: bool


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
) -> PredictionExplanationSupport:
    """Replay existing cohort/graph producers once, without scoring outputs.

    Call after scoring succeeds. ``output_dir`` is the run's ``xai`` directory;
    producer outputs use the ``pred`` stage. A fresh, unsharded trainer reads
    exactly the prediction task bundle and published checkpoint used to score.
    Errors propagate while leaving scoring artifacts and partial XAI intact.
    """
    manifest = DiskgraphTaskManifest.load(materialized.manifest_path)
    task_type = TaskType(manifest.task_type)
    cohorts = task_type in ColumnAnalysis.supported_task_types
    samples = task_type in SubgraphCollector.supported_task_types
    if not cohorts and not samples:
        return PredictionExplanationSupport(False, False)
    task = ResolvedNumericalTask.from_manifest(
        manifest, name=run_id, **(task_options or {}))
    # The published directory is already the selected fit/refit. Do not select
    # a sibling refit from its historical configuration during replay.
    trainer = DiskgraphNestedTrainer.load(
        task,
        model_dir,
        diskgraph_task_dataset_bundle_dir=materialized.bundle_dir,
        encoders=encoders if encoders is not None else load_encoders(model_dir),
        infer_config_path=False,
        diskgraph_runtime=LocalDiskgraphTrainingRuntime(),
        explain_runtime=LocalExplainRuntime(),
        embedding_models=embedding_models,
        accelerator=accelerator,
        binary_threshold=(binary_threshold
                          if binary_threshold is not None else 0.5),
    )
    if task_type.is_link_pred:
        samples = trainer.module_kwargs.module in {
            LinkPredModuleType.HYBRID,
            LinkPredModuleType.ID,
            LinkPredModuleType.SHALLOW_RHS,
        }
    support = PredictionExplanationSupport(cohorts, samples)
    methods: list[ExplainMethod] = []
    if cohorts:
        methods.append(ExplainMethod.COLUMN_ANALYSIS)
    if samples:
        methods.append(ExplainMethod.SUBGRAPH_COLLECTOR)
    if not methods:
        return support
    trainer._prediction_mode = True
    trainer.fit_mode = FitMode.fit
    trainer.init(task_manifest=manifest)
    trainer.explain(
        methods,
        stage=TrainingStage.PRED,
        fit_mode_to_use=FitMode.fit,
        root_dir=output_dir,
        top_k=top_k,
        collector_formats=['nested_json'],
    )
    # Existing producers can log and swallow errors. Missing completed output
    # must remain a failed generation, rather than silently implying success.
    required: list[str] = []
    if cohorts:
        required.append(posixpath.join(
            output_dir, 'column_analysis', 'pred', 'column_analysis.json'))
    if samples:
        collector = posixpath.join(output_dir, 'subgraph_collector', 'pred')
        mapping_path = posixpath.join(collector, 'strategy_mappings.json')
        if not filesystem.exists(mapping_path):
            raise RuntimeError('Prediction selection generation incomplete')
        mappings = load_json(mapping_path, EntityMappings)
        graphs = posixpath.join(collector, 'subgraph_json')
        ids = {artifact for group in mappings.mappings.values()
               for artifact in group}
        if not ids or any(not filesystem.exists(posixpath.join(
                graphs, f'{artifact}.json')) for artifact in ids):
            raise RuntimeError('Prediction sampled graph generation incomplete')
    if any(not filesystem.exists(path) for path in required):
        raise RuntimeError('Prediction explanation generation incomplete')
    return support
