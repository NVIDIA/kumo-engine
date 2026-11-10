# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit read contracts exercised through the mocked DiskGraph backend."""

from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper
from kumo_core.features import reads
from kumo_core.features.reads import (
    DiskgraphFeatureContext,
    read_node_times,
    read_primary_keys,
    read_tensor_frame,
)
from kumo_core.features.stores.feature import FeatureStoreTableSchema
from tests.support.diskgraph_feature_store import make_feature_context
from torch_frame import TensorFrame


@pytest.fixture
def store() -> DiskgraphFeatureContext:
    """Return four indexed rows with numerical features, keys, and timestamps."""
    return make_feature_context(
        {
            "items": pa.table(
                {
                    "id": [0, 10, 20, 30],
                    "value": [20.0, 30.0, 40.0, 50.0],
                    "created_at": pa.array([0, 10, 20, 30], type=pa.timestamp("s")),
                }
            )
        },
        {
            "items": FeatureStoreTableSchema(
                mapper_dict={"value": NumericalTensorMapper()},
                pkey_name="id",
                time_col_name="created_at",
            )
        },
        entity_table_names=["items"],
    )


def test_graph_only_metadata(store: DiskgraphFeatureContext) -> None:
    """Native metadata contains no training tensors or materialized graph data."""
    assert store.schema.num_nodes_dict == {"items": 4}
    assert store.get_pkey_name("items") == "id"
    assert store.train_table_name is None
    assert not hasattr(store, "_data")
    assert store.time_col_dict == {"items": "created_at"}


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("attribute", ["tf", "pkey", "time"])
def test_bounded_reads(
    store: DiskgraphFeatureContext,
    batch_size: int,
    attribute: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native chunking preserves all feature, key, and timestamp rows."""
    readers = {
        "tf": read_tensor_frame,
        "pkey": read_primary_keys,
        "time": read_node_times,
    }
    reader = readers[attribute]
    expected = reader(store, "items", torch.arange(4))
    assert store._engine is not None
    store._engine.fetch_features.reset_mock()
    monkeypatch.setattr(reads, "READ_BATCH_SIZE", batch_size)

    result = reader(store, "items", torch.arange(4))

    assert store._engine.fetch_features.call_count == 2
    if isinstance(result, TensorFrame):
        assert result == expected
    elif isinstance(result, torch.Tensor):
        torch.testing.assert_close(result, expected)
    else:
        np.testing.assert_array_equal(result, expected)


def test_native_read_failure_propagates(store: DiskgraphFeatureContext) -> None:
    """Local engine failures are surfaced without returning partial data."""
    assert store._engine is not None
    store._engine.fetch_features.side_effect = RuntimeError("request failed")
    with pytest.raises(RuntimeError, match="request failed"):
        read_primary_keys(store, "items", torch.arange(4))


@pytest.mark.parametrize(
    "reader", [read_tensor_frame, read_primary_keys, read_node_times]
)
def test_unknown_table_raises(
    store: DiskgraphFeatureContext,
    reader: Callable[[DiskgraphFeatureContext, str, torch.Tensor], object],
) -> None:
    """Unknown tables fail before querying the native engine."""
    with pytest.raises(KeyError):
        reader(store, "missing", torch.tensor([0]))


def test_ordered_key_and_time_reads(store: DiskgraphFeatureContext) -> None:
    """Explicit reads retain the same requested order across attributes."""
    ids = torch.tensor([3, 1])
    keys = read_primary_keys(store, "items", ids)
    timestamps = read_node_times(store, "items", ids)
    np.testing.assert_array_equal(keys, [30, 10])
    torch.testing.assert_close(timestamps, torch.tensor([30, 10]) * 1_000_000_000)
