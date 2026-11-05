# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_core.train import diskgraph_nested_trainer as trainer
from torch_geometric.sampler.base import NumNeighbors


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
