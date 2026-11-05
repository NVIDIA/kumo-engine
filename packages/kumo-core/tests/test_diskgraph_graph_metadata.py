# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema metadata replaces the DiskGraph graph-store adapter."""

from unittest.mock import mock_open, patch

import orjson
import pytest
from kumo_core.diskgraph.metadata import DiskgraphSchema, load_diskgraph_schema
from kumo_core.diskgraph.stores import DiskgraphFeatureStore
from kumo_core.encoders.config import EncodersConfig
from kumo_core.training.runtime import LocalDiskgraphTrainingRuntime


@pytest.fixture
def metadata() -> bytes:
    """Return snapshot metadata with two foreign keys and their reverse edges."""
    relations = [
        ("orders", "orders_user_id_to_users_id", "users"),
        ("users", "orders_user_id_to_users_id_reverse", "orders"),
        ("orders", "orders_product_id_to_products_id", "products"),
        ("products", "orders_product_id_to_products_id_reverse", "orders"),
    ]
    return orjson.dumps(
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
    metadata: bytes,
    table_name: str,
    expected: list[tuple[str, str]],
) -> None:
    """Explanation sees normalized outgoing keys and excludes reverse edges."""
    with patch("builtins.open", mock_open(read_data=metadata)):
        schema = load_diskgraph_schema("/snapshot")
    assert schema.get_foreign_keys(table_name) == expected
    assert ("orders", "user_id", "users") in schema.edge_types
    assert ("users", "rev_user_id", "orders") in schema.edge_types


def test_runtime_builds_feature_access_and_schema(metadata: bytes) -> None:
    """RHS/explanation setup returns usable schema metadata without a sampler."""
    with patch("builtins.open", mock_open(read_data=metadata)):
        feature_store, schema = (
            LocalDiskgraphTrainingRuntime().create_feature_store_and_schema(
                index_path="/snapshot",
                encoders=EncodersConfig(),
                entity_table_names=["users", "products"],
                time_col_dict={},
            )
        )
    assert isinstance(feature_store, DiskgraphFeatureStore)
    assert isinstance(schema, DiskgraphSchema)
    assert feature_store._engine is None
    assert feature_store.entity_table_names == ["users", "products"]
    assert feature_store.get_tensor_size("orders", "tf") == (3,)
    assert schema.get_foreign_keys("orders") == [
        ("user_id", "users"),
        ("product_id", "products"),
    ]


def test_collector_recovers_foreign_keys_from_schema(metadata: bytes) -> None:
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
        feature_store, schema = (
            LocalDiskgraphTrainingRuntime().create_feature_store_and_schema(
                index_path="/snapshot",
                encoders=EncodersConfig(),
                entity_table_names=["users"],
                time_col_dict={},
            )
        )
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
