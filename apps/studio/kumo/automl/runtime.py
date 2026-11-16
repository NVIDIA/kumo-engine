# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monolith environment adapters for the portable AutoTrainer."""

from __future__ import annotations

import contextlib
import os.path as osp
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ContextManager

from kumo_core.artifacts import VitalArtifactSources
from kumo_core.automl.runtime import (
    AutoTrainingRuntime,
    collect_resolved_encoders,
)
from kumo_core.automl.target_fanout import infer_target_edge_fanout
from kumo_core.instrumentation import TrainingInstrumentation
from kumo_core.resolved_task import ResolvedNumericalTask
from kumo_core.types import TrainTableType

from common_utils.kumo_timing import kumo_timing
from kumo.home import get_home_dir
from kumo.metrics import MetricsClient
from kumo.task.task import Task
from kumo.train.config_io import load_job_automl_config
from kumo.train.trainer_runtime import monolith_trainer_runtime_kwargs


def infer_authoring_max_target_neighbors_per_entity(task: Task) -> int | None:
    """Resolve the target-edge fanout from the authoring task's own data.

    The degree policy is `kumo_core`'s, shared with portable training; what
    the monolith adds is where the two numbers come from. `Task` and
    `Graph` do not exist in `kumo_core`, so they stay behind these two
    closures rather than crossing the boundary.
    """
    def avg_entity_degree() -> float | None:
        train_table_stats = task.get_train_table_stats(after_split=False)
        temporal_types = {
            TrainTableType.TEMPORAL_NODE_PRED,
            TrainTableType.TEMPORAL_LINK_PRED,
        }
        if (train_table_stats is not None
                and train_table_stats.stats_type in temporal_types and getattr(
                    train_table_stats, 'node_degree_stats', None) is not None):
            return train_table_stats.node_degree_stats.avg_entity_degree
        return None

    return infer_target_edge_fanout(
        task.target_to_entity_paths,
        row_count=lambda table_name: task.graph[table_name].num_rows(
            compute_table_stats_if_missing=True),
        avg_entity_degree=avg_entity_degree,
    )


class MonolithAutoTrainingRuntime(AutoTrainingRuntime):
    def default_home_dir(self) -> str:
        return get_home_dir()

    def metrics_job_name(self) -> str:
        return MetricsClient().job_name

    def collect_encoders(self, task: ResolvedNumericalTask):
        return collect_resolved_encoders(task)

    def load_task(self, task_name: str, home_dir: str) -> Any:
        return Task.load(task_name, home_dir)

    def load_automl_config(self, job_config: Any) -> Any:
        return load_job_automl_config(job_config)

    def trainer_runtime_kwargs(self) -> dict[str, Any]:
        return monolith_trainer_runtime_kwargs()


class MonolithAutoTrainingInstrumentation(TrainingInstrumentation):
    def record_metrics_job_name(self, job_name: str) -> None:
        return None

    @contextlib.contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        yield

    def timing(self, name: str, **fields: Any) -> ContextManager[None]:
        return kumo_timing(name, **fields)


def monolith_vital_artifact_sources(
    training_input: Any,
    auto_trainer: Any,
    best_trainer: Any,
    metrics: Any,
) -> VitalArtifactSources:
    """Describe the established monolith artifact locations without copying."""
    bundle_dir = training_input.task_dataset_bundle_dir
    trainer_dir = best_trainer.log_dir
    return VitalArtifactSources(
        selected_checkpoint=Path(best_trainer.best_ckpt),
        trainer_config=Path(
            osp.join(best_trainer.original_log_dir, 'config.json')),
        encoders=Path(osp.join(auto_trainer.log_dir, 'encoders.json')),
        target_encoder=Path(osp.join(trainer_dir, 'target_encoder.json')),
        resolved_model_plan=Path(osp.join(trainer_dir, 'model_plan.json')),
        graph_manifest=Path(osp.join(bundle_dir, 'diskgraph_manifest.json')),
        task_manifest=Path(osp.join(bundle_dir, 'manifest.json')),
        input_manifest=Path(osp.join(training_input.home_dir, 'task.json')),
        metrics=Path(osp.join(trainer_dir, 'metrics.json')),
        prediction_schema=Path(osp.join(trainer_dir, 'prediction_schema.json')),
    )


class MonolithVitalArtifactPublisher:
    """Expose monolith-owned artifact paths; legacy deployment owns copying."""
    def publish(self, sources: VitalArtifactSources) -> dict[str, str]:
        return {name: str(path) for name, path in sources.as_dict().items()}


__all__ = [
    'MonolithAutoTrainingInstrumentation',
    'MonolithAutoTrainingRuntime',
    'MonolithVitalArtifactPublisher',
    'infer_authoring_max_target_neighbors_per_entity',
    'monolith_vital_artifact_sources',
]
