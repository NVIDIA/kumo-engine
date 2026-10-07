# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment capabilities consumed by the portable Trainer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Tuple

import psutil
from diskgraph import SamplerEngine
from kumo_core.diskgraph.stores import (
    DiskgraphFeatureStore,
    DiskgraphGraphStore,
    diskgraph_auto_config,
    diskgraph_index_size_bytes,
)

if TYPE_CHECKING:
    from kumo_core.embedding_models import EmbeddingModelStore


class TrainingDataRuntime(Protocol):
    def create_graph_sampler(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def create_datamodule(self, *args: Any, **kwargs: Any) -> Any:
        ...


class TrainingArtifactRuntime(Protocol):
    def default_model_root(self) -> str:
        ...

    def local_model_root(self) -> str:
        ...


class DiskgraphTrainingRuntime(Protocol):
    """Engine capabilities needed by DiskGraph-native numerical training."""
    def auto_config(
        self,
        *,
        index_path: str,
        estimated_edges: int,
    ) -> dict[str, Any]:
        ...

    def create_feature_and_graph_stores(
        self,
        *,
        index_path: str,
        encoders: Any,
        entity_table_names: list[str],
        time_col_dict: dict[str, str],
    ) -> tuple[Any, Any]:
        ...


class LocalDiskgraphTrainingRuntime:
    """Construct core DiskGraph stores for a driver-local snapshot."""
    def auto_config(
        self,
        *,
        index_path: str,
        estimated_edges: int,
    ) -> dict[str, Any]:
        return diskgraph_auto_config(
            cpu_count=os.cpu_count() or 1,
            ram_bytes=psutil.virtual_memory().total,
            index_bytes=diskgraph_index_size_bytes(index_path),
            estimated_edges=estimated_edges,
        )

    def create_feature_and_graph_stores(
        self,
        *,
        index_path: str,
        encoders: Any,
        entity_table_names: list[str],
        time_col_dict: dict[str, str],
    ) -> tuple[Any, Any]:
        feature_store = DiskgraphFeatureStore.for_explain(
            index_path=index_path,
            encoders=encoders,
            entity_table_names=entity_table_names,
            time_col_dict=time_col_dict,
        )
        graph_store = DiskgraphGraphStore(index_path=index_path)
        feature_store.set_sampler_engine(SamplerEngine(index_path))
        return feature_store, graph_store


class LocalTrainingArtifactRuntime:
    r"""Resolves every Trainer path under one run directory."""
    def __init__(self, model_root: str) -> None:
        self.model_root = str(Path(model_root))

    def default_model_root(self) -> str:
        return self.model_root

    def local_model_root(self) -> str:
        return self.model_root


class LocalExplainRuntime:
    r"""Replays a trained model over the DiskGraph Lightning data module."""

    prediction_selection_nodes = 1000
    max_batches = 50
    max_facts = 16
    mapping_file_name = 'strategy_mappings.json'

    def get_explain_node_inputs(self, datamodule: Any,
                                stage: Any) -> Tuple[Any, Any]:
        _, input_nodes, _, input_id = datamodule.stage_inputs(stage.value)
        return input_nodes, input_id

    def select_input_nodes(self, *args: Any, **kwargs: Any) -> Any:
        from kumo_core.training.explain_selection import select_input_nodes

        return select_input_nodes(*args, **kwargs)

    def get_eval_dataloader(self, datamodule: Any, stage: Any,
                            seed_nodes: Any) -> Any:
        return datamodule.explain_dataloader(stage.value, seed_nodes)


class LocalAutoTrainingRuntime:
    r"""Supplies ``kumo_core``'s AutoTrainer with one resolved local run.

    The task is resolved before training starts, so the ``load_*`` methods hand
    back the one task this run was built for rather than reading it from disk.
    They still check what they are asked for: a mismatch means the caller is
    mixing runs, which would train against the wrong data.
    """
    def __init__(
        self,
        *,
        home_dir: str,
        resolved_task: Any,
        run_name: Optional[str] = None,
        embedding_models: Optional[EmbeddingModelStore] = None,
    ) -> None:
        self.home_dir = str(Path(home_dir))
        self.resolved_task = resolved_task
        self.run_name = run_name
        self.embedding_models = embedding_models
        self.artifact_runtime = LocalTrainingArtifactRuntime(self.home_dir)
        self.diskgraph_runtime = LocalDiskgraphTrainingRuntime()
        self.explain_runtime = LocalExplainRuntime()

    def default_home_dir(self) -> str:
        return self.home_dir

    def metrics_job_name(self) -> str:
        return str(self.run_name or 'kumo-engine')

    def collect_encoders(self, task: Any) -> Any:
        from kumo_core.automl.runtime import collect_resolved_encoders

        if (task is not self.resolved_task
                and task.name != self.resolved_task.name):
            raise ValueError('encoder artifact does not belong to this task')
        return collect_resolved_encoders(task)

    def load_task(self, task_name: str, home_dir: str) -> Any:
        if task_name != self.resolved_task.name or home_dir != self.home_dir:
            raise ValueError('requested task does not match resolved input')
        return self.resolved_task

    def load_task_from_home(self, home_dir: str) -> Any:
        if home_dir != self.home_dir:
            raise ValueError('requested home does not match resolved input')
        return self.resolved_task

    def load_automl_config(self, job_config: Any) -> Any:
        raise RuntimeError(
            'The local backend trains from a resolved model plan; there is no '
            'AutoML config file to load.')

    def trainer_runtime_kwargs(self) -> Dict[str, Any]:
        return {
            'artifact_runtime': self.artifact_runtime,
            'diskgraph_runtime': self.diskgraph_runtime,
            'explain_runtime': self.explain_runtime,
            'embedding_models': self.embedding_models,
        }


def _trainer_validation_metrics(trainer: Any) -> Dict[str, float]:
    return {
        name: value
        for name, value in trainer.get_train_results().items()
        if name.startswith('val_')
    }


def local_training_bindings(
    training_input: Any,
    run_name: Optional[str] = None,
    embedding_models: Optional[EmbeddingModelStore] = None,
) -> Any:
    r"""Binds one resolved training input to this process.

    Both local SDK entry points use the same runtime and publication policy.
    """
    from kumo_core.artifacts import (
        FilesystemVitalArtifactPublisher,
        resolve_vital_artifact_sources,
    )
    from kumo_core.instrumentation import NoOpTrainingInstrumentation
    from kumo_core.training.orchestration import TrainingBindings

    return TrainingBindings(
        runtime=LocalAutoTrainingRuntime(home_dir=training_input.home_dir,
                                         resolved_task=training_input.task,
                                         run_name=run_name,
                                         embedding_models=embedding_models),
        instrumentation=NoOpTrainingInstrumentation(),
        resolve_vital_artifacts=resolve_vital_artifact_sources,
        # The vital bundle lands beside the run. A platform backend would then
        # upload it to a registry; locally the path is the identity.
        artifact_publisher=FilesystemVitalArtifactPublisher(
            Path(training_input.home_dir) / 'published-model'),
        load_validation_metrics=_trainer_validation_metrics,
    )
