# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep predicted items explicit when graph export aggregates neighbors."""

from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pytest
import torch
from kumo_core.explain import subgraph_collector
from kumo_core.explain.subgraph_collector import SubgraphCollector
from kumoapi.explain.subgraph import SubgraphNode
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from torch_geometric.data import HeteroData


@pytest.mark.parametrize("max_num_facts", [0, 1, 2])
def test_export_retains_predicted_items(
    max_num_facts: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retain each predicted key once, including aggregated and unsampled items.

    Args:
        max_num_facts: Number of neighbors retained before aggregation.
        monkeypatch: Fixture for replacing filesystem writes.
        tmp_path: Output directory supplied to the collector.
    """
    batch = HeteroData()
    batch["users"].input_id = torch.tensor([0])
    batch["users"].num_nodes = 1
    batch["users"].batch_size = 1
    batch["users"].batch = torch.tensor([0])
    batch["users"].num_sampled_nodes = [1, 0]
    batch["items"].num_nodes = 2
    batch["items"].batch = torch.tensor([0, 0])
    batch["items"].n_id = torch.tensor([7, 8])
    batch["items"].num_sampled_nodes = [0, 2]
    edge = batch["users", "item", "items"]
    edge.edge_index = torch.tensor([[0, 0], [0, 1]])
    edge.num_sampled_edges = [2]
    feature_store = Mock()
    feature_store.get_pkey_name.return_value = "ID"
    feature_store.get_time_col_name.return_value = None
    graph_schema = Mock()
    graph_schema.get_foreign_keys.return_value = []
    rhs_index_map = Mock()
    rhs_index_map.loc = {7: {"ID": 107}, 8: {"ID": 108}, 9: {"ID": 109}}
    save_json = Mock()
    monkeypatch.setattr(subgraph_collector, "save_json", save_json)
    monkeypatch.setattr(subgraph_collector.fsutil, "makedirs", Mock())
    collector = SubgraphCollector(
        TaskType.STATIC_LINK_PREDICTION,
        str(tmp_path),
        ["nested_json"],
        max_num_facts=max_num_facts,
    )

    collector.format_subgraph_json(
        batch,
        [1000],
        {
            "users": pa.table({"ID": [1000]}).to_pandas(),
            "items": pa.table({"ID": [107, 108]}).to_pandas(),
        },
        {},
        TrainingStage.PRED,
        feature_store,
        graph_schema,
        Mock(entity_table_names=["users", "items"]),
        rhs_index_map,
        torch.tensor([[7, 8, 9]]),
        pa.table({"ID": [107, 108, 109]}).to_pandas(),
    )

    store = save_json.call_args_list[-1].args[0].subgraph_store
    item_keys = [
        store.nodes[node_id].pkey
        for node_id in store.tables["items"]
        if isinstance(store.nodes[node_id], SubgraphNode)
    ]
    assert sorted(item_keys) == [107, 108, 109]
