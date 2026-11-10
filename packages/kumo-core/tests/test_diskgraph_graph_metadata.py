# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema metadata replaces the DiskGraph graph-store adapter."""

import json
from unittest.mock import mock_open, patch

import pytest
from kumo_core.diskgraph.metadata import DiskgraphSchema, load_diskgraph_schema
from kumo_core.encoders.config import EncodersConfig
from kumo_core.features.reads import DiskgraphFeatureContext
from kumo_core.training.runtime import LocalDiskgraphTrainingRuntime


@pytest.fixture
def metadata() -> str:
    """Return snapshot metadata with two foreign keys and their reverse edges."""
    relations = [
        ("orders", "orders_user_id_to_users_id", "users"),
        ("users", "orders_user_id_to_users_id_reverse", "orders"),
        ("orders", "orders_product_id_to_products_id", "products"),
        ("products", "orders_product_id_to_products_id_reverse", "orders"),
    ]
    return json.dumps(
        {
            "node_types": [
                {"name": name, "max_id": 2, "primary_key_map": {"column": "id"}}
                for name in ("orders", "users", "products")
            ],
            "edge_relations": [
                {"src_type": source, "name": relation, "dst_type": destination}
                for source, relation, destination in relations
            ],
        }
    )


@pytest.mark.parametrize(
    "table_name, expected",
    [
        ("orders", [("user_id", "users"), ("product_id", "products")]),
        ("users", []),
        ("products", []),
        ("missing", []),
    ],
)
def test_foreign_keys_from_snapshot_metadata(
    metadata: str,
    table_name: str,
    expected: list[tuple[str, str]],
) -> None:
    """Explanation sees normalized outgoing keys and excludes reverse edges."""
    with patch("builtins.open", mock_open(read_data=metadata)):
        schema = load_diskgraph_schema("/snapshot")
    assert schema.get_foreign_keys(table_name) == expected
    assert ("orders", "user_id", "users") in schema.edge_types
    assert ("users", "rev_user_id", "orders") in schema.edge_types


@pytest.mark.parametrize(
    "source_table, column, destination_table",
    [
        ("orders", "user_id", "users"),
        ("orders", "rev_user_id", "users"),
        ("catalog.orders", "rev_user_id", "catalog.users"),
    ],
)
def test_foreign_keys_use_parsed_forward_specs(
    source_table: str,
    column: str,
    destination_table: str,
) -> None:
    """Keep rev_-prefixed forward keys and exclude unparsed and reverse edges."""
    forward_relation = f"{source_table}_{column}_to_{destination_table}_id"
    relations = [
        (source_table, forward_relation, destination_table),
        (destination_table, f"{forward_relation}_reverse", source_table),
        (source_table, "unparsed_relation", destination_table),
        (destination_table, "unparsed_relation_reverse", source_table),
    ]
    metadata = json.dumps(
        {
            "edge_relations": [
                {"src_type": source, "name": relation, "dst_type": destination}
                for source, relation, destination in relations
            ],
        }
    )

    with patch("builtins.open", mock_open(read_data=metadata)):
        schema = load_diskgraph_schema("/snapshot")

    assert schema.get_foreign_keys(source_table) == [(column, destination_table)]
    assert schema.get_foreign_keys(destination_table) == []
    assert schema.get_foreign_keys("missing") == []
    assert (source_table, "unparsed_relation", destination_table) in schema.edge_types


def test_runtime_builds_feature_access_and_schema(metadata: str) -> None:
    """RHS/explanation setup returns usable schema metadata without a sampler."""
    with patch("builtins.open", mock_open(read_data=metadata)) as open_metadata:
        feature_store = (
            LocalDiskgraphTrainingRuntime().create_feature_context(
                index_path="/snapshot",
                encoders=EncodersConfig(),
                entity_table_names=["users", "products"],
                time_col_dict={},
            )
        )
    schema = feature_store.schema
    open_metadata.assert_called_once()
    assert schema.num_nodes_dict == {"orders": 3, "users": 3, "products": 3}
    assert isinstance(feature_store, DiskgraphFeatureContext)
    assert isinstance(schema, DiskgraphSchema)
    assert feature_store._engine is None
    assert feature_store.entity_table_names == ["users", "products"]
    assert schema.num_nodes_dict["orders"] == 3
    assert schema.get_foreign_keys("orders") == [
        ("user_id", "users"),
        ("product_id", "products"),
    ]


def test_collector_recovers_foreign_keys_from_schema(metadata: str) -> None:
    """Explanation reconstructs foreign-key values using schema and batch edges."""
    from types import SimpleNamespace

    import numpy as np
    import torch
    from kumo_core.explain import SubgraphCollector
    from kumo_core.types import TaskType
    from kumoapi.train import TrainingStage
    from torch_frame import TensorFrame
    from torch_geometric.data import HeteroData

    with patch("builtins.open", mock_open(read_data=metadata)):
        feature_store = (
            LocalDiskgraphTrainingRuntime().create_feature_context(
                index_path="/snapshot",
                encoders=EncodersConfig(),
                entity_table_names=["users"],
                time_col_dict={},
            )
        )
    schema = feature_store.schema
    batch = HeteroData()
    for name, key in [("users", 42), ("orders", 100), ("products", 7)]:
        batch[name].num_nodes = 1
        batch[name].pkey = np.array([key])
        batch[name].tf = TensorFrame({}, {}, num_rows=1)
        batch[name].batch = torch.tensor([0])
    batch["users"].batch_size = 1
    batch["users"].input_id = torch.tensor([0])
    batch["orders", "user_id", "users"].edge_index = torch.tensor([[0], [0]])
    batch["orders", "product_id", "products"].edge_index = torch.tensor([[0], [0]])
    collector = SubgraphCollector(
        TaskType.REGRESSION, "/mock-explanation", formats=["csv"]
    )
    with patch.object(collector, "format_csv") as write:
        collector.explain_post_step(
            batch,
            torch.nn.Identity(),
            SimpleNamespace(data=(feature_store, schema)),
            {},
            TrainingStage.TEST,
            0,
        )
    write.assert_called_once()
    frames = write.call_args.args[2]
    assert frames["orders"]["user_id"].tolist() == [42]
    assert frames["orders"]["product_id"].tolist() == [7]
    assert feature_store._engine is None
