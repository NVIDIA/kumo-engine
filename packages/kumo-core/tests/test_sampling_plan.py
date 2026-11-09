# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.io.json import save_json
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.resolved_task import (
    ResolvedNumericalTask,
    ResolvedTaskGraph,
    ResolvedTrainTable,
)
from kumo_core.train import diskgraph_nested_trainer as trainer
from kumo_core.train.trainer import (
    Trainer,
    _resolve_num_neighbors,
    _restore_num_neighbors_key,
)
from kumo_core.training.config import SamplerKwargs
from kumo_core.types import TaskType
from torch_geometric.sampler.base import NumNeighbors
from torch_geometric.typing import EdgeTypeStr


@pytest.mark.parametrize("source", ["sales.events", "catalog.sales.events"])
def test_sampling_plan_resolves_relations_of_a_dotted_source_table(
        monkeypatch, source):
    relation = f"{source}_user_id_to_users_id"
    monkeypatch.setattr(
        trainer, "_load_index_metadata", lambda metadata: {
            "edge_relations": [{
                "name": relation,
                "src_type": source,
                "dst_type": "users",
            }, {
                "name": f"{relation}_reverse",
                "src_type": "users",
                "dst_type": source,
            }]
        })
    fake = SimpleNamespace(sampler_kwargs=SimpleNamespace(
        num_neighbors=NumNeighbors({(source, "user_id", "users"): [10, 5]}),
        temporal_strategy="last"))

    plan = trainer.DiskgraphNestedTrainer._build_ttng_sampling_plan(
        fake, SimpleNamespace(graph_runtime_metadata=None))

    assert [set(layer["fanouts"]) for layer in plan["layers"]
            ] == [{f"{relation}_reverse"}] * 2
    assert [
        layer["fanouts"][f"{relation}_reverse"]["fanout"]
        for layer in plan["layers"]
    ] == [10, 5]


@pytest.mark.parametrize("sources, resolved", [
    (["sales.events"], True),
    (["sales.events", "crm.events"], False),
])
def test_sampling_plan_accepts_a_stripped_override_only_when_unambiguous(
        monkeypatch, sources, resolved):
    monkeypatch.setattr(
        trainer, "_load_index_metadata", lambda metadata: {
            "edge_relations": [{
                "name": f"{source}_user_id_to_users_id",
                "src_type": source,
                "dst_type": "users",
            } for source in sources]
        })
    fake = SimpleNamespace(sampler_kwargs=SimpleNamespace(
        num_neighbors=NumNeighbors({("events", "user_id", "users"): [10]},
                                   default=[3]),
        temporal_strategy="last"))

    plan = trainer.DiskgraphNestedTrainer._build_ttng_sampling_plan(
        fake, SimpleNamespace(graph_runtime_metadata=None))

    assert set(plan["layers"][0]["fanouts"]) == (
        {"sales.events_user_id_to_users_id"} if resolved else set())


def test_restored_sampler_keys_keep_matching_edge_types_exactly():
    edge_types = {
        str(EdgeTypeStr(*edge_type)): edge_type
        for edge_type in [("sales.events", "user_id", "users")]
    }

    restore = lambda *key: _restore_num_neighbors_key(
        EdgeTypeStr(*key), edge_types).to_tuple()

    assert restore("sales.events", "user_id", "users") == (
        "sales.events", "user_id", "users")
    assert restore("legacy.orders", "user_id", "users") == (
        "orders", "user_id", "users")
    assert _restore_num_neighbors_key(
        EdgeTypeStr("crm.events", "user_id", "users"), {}).to_tuple() == (
            "crm.events", "user_id", "users")


def test_plain_sampler_path_recovers_a_legacy_key_loaded_without_a_graph(
        tmp_path, monkeypatch):
    task = ResolvedNumericalTask(
        name="test",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metrics=["auroc"],
        graph=ResolvedTaskGraph(node_types=("users", ), edge_types=(),
                                seed_node_type="users",
                                target_node_type="users"),
        train_table=ResolvedTrainTable(name="train_table",
                                       target_column="TARGET",
                                       time_column=None, weight_column=None),
    )
    saved = Trainer(
        task=task,
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        root_dir=str(tmp_path),
        sampler_kwargs=SamplerKwargs(num_neighbors=NumNeighbors(
            {("ds.orders", "user_id", "ds.users"): [10, 5]}, default=[3, 3])),
    )
    save_json(saved.config(), str(tmp_path / "config.json"))
    fanouts = {}

    class Sampled(Exception):
        pass

    def create_graph_sampler(feature_store, graph_store, sampler_kwargs,
                             **kwargs):
        fanouts.update(
            sampler_kwargs.num_neighbors.get_values(graph_store.edge_types))
        raise Sampled

    loaded = Trainer.load(
        task, str(tmp_path), infer_config_path=False,
        data_runtime=SimpleNamespace(create_graph_sampler=create_graph_sampler))

    assert loaded.sampler_kwargs.num_neighbors.values == {
        EdgeTypeStr("ds.orders", "user_id", "ds.users"): [10, 5]
    }

    loaded.feature_store = SimpleNamespace(
        table_names=["orders", "users"], train_table_name="train_table",
        has_train_table_time_col=False, time_col_dict={},
        entity_table_names=["users"])
    loaded.graph_store = SimpleNamespace(edge_types=[
        ("orders", "user_id", "users"), ("users", "rev_user_id", "orders")
    ])
    monkeypatch.setattr(Trainer, "initialize_model",
                        lambda self, **kwargs: None)
    with pytest.raises(Sampled):
        loaded.init()

    assert fanouts == {
        ("orders", "user_id", "users"): [10, 5],
        ("users", "rev_user_id", "orders"): [3, 3],
    }


def test_resolved_sampler_keys_keep_an_exact_fanout_over_a_legacy_alias():
    exact = ("orders", "user_id", "users")
    alias = ("ds.orders", "user_id", "ds.users")
    for ordered in ([exact, alias], [alias, exact]):
        fanouts = {key: [10] if key == exact else [3] for key in ordered}

        resolved = _resolve_num_neighbors(NumNeighbors(fanouts, default=[1]),
                                          [exact])

        assert resolved.get_values([exact]) == {exact: [10]}
