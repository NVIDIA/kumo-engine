# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for trainer-owned feature access and schema setup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.io.json import from_json_dict, to_json_dict
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.resolved_task import (
    ResolvedNumericalTask,
    ResolvedTaskGraph,
    ResolvedTrainTable,
)
from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
from kumo_core.train.trainer import Trainer
from kumo_core.training.config import TrainerConfig
from kumo_core.training.task_config import TaskConfig
from kumo_core.types import TaskType


@pytest.fixture
def task() -> ResolvedNumericalTask:
    """Return a binary task with graph metadata for loading tests."""
    return ResolvedNumericalTask(
        name="store-lifecycle",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metrics=["auroc"],
        graph=ResolvedTaskGraph(
            node_types=("users", "orders"),
            edge_types=(("orders", "USER_ID_", "users"),),
            seed_node_type="users",
            target_node_type="users",
        ),
        train_table=ResolvedTrainTable(
            name="train_table",
            target_column="TARGET",
            time_column=None,
            weight_column=None,
        ),
    )


@pytest.mark.parametrize("trainer_class", [Trainer, DiskgraphNestedTrainer])
@pytest.mark.parametrize("name", ["feature_store", "graph_store"])
@pytest.mark.parametrize("value", [None, object()])
def test_constructor_rejects_store_injection(
    task: ResolvedNumericalTask,
    trainer_class: type[Trainer],
    name: str,
    value: object,
) -> None:
    """Reject retired store arguments, including explicit None values."""
    with pytest.raises(TypeError, match=f"{name} is managed internally"):
        trainer_class(
            task,
            diskgraph_task_dataset_bundle_dir="bundle",
            **{name: value},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("trainer_class", [Trainer, DiskgraphNestedTrainer])
@pytest.mark.parametrize("name", ["feature_store", "graph_store"])
@pytest.mark.parametrize("value", [None, object()])
def test_load_rejects_store_injection_before_reading_config(
    monkeypatch: pytest.MonkeyPatch,
    task: ResolvedNumericalTask,
    trainer_class: type[Trainer],
    name: str,
    value: object,
) -> None:
    """Reject retired loading arguments before accessing model artifacts."""
    load_config = Mock()
    monkeypatch.setattr("kumo_core.train.trainer.load_json", load_config)

    with pytest.raises(TypeError, match=f"{name} is managed internally"):
        trainer_class.load(
            task,
            "model",
            diskgraph_task_dataset_bundle_dir="bundle",
            **{name: value},  # type: ignore[arg-type]
        )

    load_config.assert_not_called()


def test_load_recovers_legacy_edge_names_from_task(
    monkeypatch: pytest.MonkeyPatch,
    task: ResolvedNumericalTask,
) -> None:
    """Recover trailing underscores without injecting schema or feature stores."""
    trainer = Trainer(
        task,
        root_dir="model",
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=1),
        head=NodePredHead(),
        sampler_kwargs={"num_neighbors": {"orders__USER_ID___users": [10]}},
    )
    config = from_json_dict(to_json_dict(trainer.config()), TrainerConfig)
    monkeypatch.setattr(
        "kumo_core.train.trainer.load_json",
        Mock(return_value=config),
    )

    loaded = DiskgraphNestedTrainer.load(
        task,
        "model",
        version=None,
        infer_config_path=False,
        diskgraph_task_dataset_bundle_dir="bundle",
    )

    assert loaded.feature_store is None
    assert loaded.graph_schema is None
    assert not hasattr(loaded, "graph_store")
    assert {
        edge.to_tuple() for edge in loaded.sampler_kwargs.num_neighbors.values
    } == set(task.graph.edge_types)


def test_explain_creates_feature_store_and_schema_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create both resources once and restore data after repeated explanations."""
    trainer = DiskgraphNestedTrainer.__new__(DiskgraphNestedTrainer)
    trainer.feature_store = None
    trainer.diskgraph_task_dataset_bundle_dir = "bundle"
    trainer.encoders = None
    previous_data = object()
    trainer.datamodule = SimpleNamespace(_data=previous_data, time_col_dict={})
    feature_store, schema = Mock(), Mock()
    runtime = Mock()
    runtime.create_feature_store_and_schema.return_value = feature_store, schema
    trainer.diskgraph_runtime = runtime
    manifest = SimpleNamespace(
        graph_runtime_metadata={"graph_index_path": "index"},
        metadata={"entity_table_name": "users"},
        seed_node_type="users",
    )
    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer.DiskgraphTaskManifest.load",
        Mock(return_value=manifest),
    )
    observed: list[object] = []
    parent_explain = Mock(
        side_effect=lambda: observed.append(trainer.datamodule._data),
    )
    monkeypatch.setattr(Trainer, "explain", parent_explain)

    trainer.explain()
    trainer.explain()

    assert trainer.feature_store is feature_store
    assert trainer.graph_schema is schema
    assert observed == [(feature_store, schema)] * 2
    assert trainer.datamodule._data is previous_data
    runtime.create_feature_store_and_schema.assert_called_once_with(
        index_path="index",
        encoders=None,
        entity_table_names=["users"],
        time_col_dict={},
    )


@pytest.mark.parametrize("initialized", [False, True])
def test_base_trainer_rejects_legacy_initialization(initialized: bool) -> None:
    """The retired base path cannot try to sample graphs or read train tensors."""
    trainer = Trainer.__new__(Trainer)
    trainer._fully_initialized = initialized
    with pytest.raises(NotImplementedError, match="Use DiskgraphNestedTrainer"):
        trainer.init()


def test_load_accepts_task_config_without_graph(
    monkeypatch: pytest.MonkeyPatch,
    task: ResolvedNumericalTask,
) -> None:
    """Service result loading needs no graph metadata to read trainer config."""
    trainer = Trainer(
        task,
        root_dir="model",
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=1),
        head=NodePredHead(),
        sampler_kwargs={"num_neighbors": {"orders__user_id__users": [10]}},
    )
    config = from_json_dict(to_json_dict(trainer.config()), TrainerConfig)
    monkeypatch.setattr(
        "kumo_core.train.trainer.load_json",
        Mock(return_value=config),
    )
    task_config = TaskConfig(task_type=task.task_type, name=task.name)

    loaded = Trainer.load(
        task_config,
        "model",
        version=None,
        infer_config_path=False,
    )

    assert loaded.task is task_config
    assert loaded.graph_schema is None
    assert not isinstance(loaded.sampler_kwargs, dict)
    assert {edge.to_tuple() for edge in loaded.sampler_kwargs.num_neighbors.values} == {
        ("orders", "user_id", "users")
    }
