# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service fixture metadata remains usable after retiring in-memory stores."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.metadata import DiskgraphSchema
from kumo_core.types import TaskType
from kumoapi.typing import Stype


@pytest.fixture
def task_fixtures(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the fixture module without unrelated service package initialization."""
    path = Path(__file__).parents[3] / "kumo/testing/fixtures/task.py"
    spec = spec_from_file_location("service_task_fixture_metadata", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("temporal", [False, True])
def test_task_configuration_metadata(
    task_fixtures: ModuleType,
    temporal: bool,
) -> None:
    """Configuration fixtures preserve encoders, graph relations, and key reads."""
    users = pa.table({"id": [20, 10, 30], "age": [20.0, 30.0, 40.0]}).to_pandas()
    orders = pa.table({"id": [100, 101], "user_id": [20, 30]})
    train = pa.table({"id": [20, 10, 30], "TARGET": [True, False, True]})
    if temporal:
        train = train.append_column("TIME", pa.array([1, 2, 3], pa.timestamp("s")))
        orders = orders.append_column("created_at", pa.array([1, 2], pa.timestamp("s")))
        orders = orders.append_column("ended_at", pa.array([3, 4], pa.timestamp("s")))
    schemas = {
        "users": task_fixtures.TestTableSchema(
            stypes={"age": Stype.numerical}, pkey_name="id"
        ),
        "orders": task_fixtures.TestTableSchema(
            fkeys=[("user_id", "users")],
            time_col="created_at" if temporal else None,
            end_time_col="ended_at" if temporal else None,
        ),
    }
    feature_store, graph_schema, encoders = task_fixtures.create_task_data(
        df_dict={"users": users, "orders": orders.to_pandas()},
        schema_dict=schemas,
        task_type=TaskType.BINARY_CLASSIFICATION,
        train_table_df=train.to_pandas(),
    )

    assert isinstance(graph_schema, DiskgraphSchema)
    assert graph_schema.get_foreign_keys("orders") == [("user_id", "users")]
    assert graph_schema.get_foreign_keys("users") == []
    assert ("users", "rev_user_id", "orders") in graph_schema.edge_types
    assert feature_store.table_names == ["users", "orders"]
    assert feature_store.train_table_name == "TRAIN_TABLE"
    assert feature_store.entity_table_names == ["users"]
    assert feature_store.train_fkey_names == ["id"]
    assert feature_store.has_train_table_time_col is temporal
    assert feature_store.get_time_col_name("TRAIN_TABLE") == (
        "TIME" if temporal else None
    )
    assert feature_store.get_time_col_name("users") is None
    assert feature_store.get_end_time_col_name("users") is None
    assert feature_store.get_end_time_col_name("TRAIN_TABLE") is None
    assert feature_store.get_time_col_name("orders") == (
        "created_at" if temporal else None
    )
    assert feature_store.get_end_time_col_name("orders") == (
        "ended_at" if temporal else None
    )
    assert feature_store.get_tensor_size("users") == (3,)
    assert feature_store.get_tensor_size("orders", "tf") == (2,)
    assert feature_store.get_tensor_size("TRAIN_TABLE") == (3,)
    assert encoders.target is not None
    assert feature_store.get_target_tensor_mapper() is not None
    assert feature_store.get_tensor_mapper("users", "age") is not None
    with feature_store.safe_batch_mode(2):
        np.testing.assert_array_equal(
            feature_store["users", "pkey", torch.tensor([2, 0])], [30, 20]
        )
    np.testing.assert_array_equal(feature_store["users", "pkey", :], [20, 10, 30])
    with pytest.raises(KeyError, match="only reads primary keys"):
        feature_store["users", "tf", torch.tensor([0])]
