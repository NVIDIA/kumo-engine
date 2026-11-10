# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared FeatureStore contracts exercised through the DiskGraph backend."""

from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.stores import DiskgraphFeatureStore
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper
from kumo_core.features.stores.feature import FeatureStoreAttr, FeatureStoreTableSchema
from kumo_core.types.features import FeatureStoreData
from tests.support.diskgraph_feature_store import make_diskgraph_feature_store
from torch_frame import TensorFrame


@pytest.fixture
def store() -> DiskgraphFeatureStore:
    """Return four indexed rows with numerical features, keys, and timestamps."""
    return make_diskgraph_feature_store(
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


def test_graph_only_metadata(store: DiskgraphFeatureStore) -> None:
    """Graph-only stores expose no training-table attributes or materialized data."""
    assert store.get_all_tensor_attrs() == [
        FeatureStoreAttr("items", "tf"),
        FeatureStoreAttr("items", "pkey"),
        FeatureStoreAttr("items", "time"),
    ]
    assert store.train_table_name is None
    assert store.train_fkey_names == []
    assert store.get_target_tensor_mapper() is None
    assert not store.has_train_table_time_col
    assert not store.has_train_table_test_mask
    assert not store.has_train_table_target_col
    assert not store.has_train_table_weight_col
    assert not hasattr(store, "_data")
    assert store.time_col_dict == {"items": "created_at"}


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("attribute", ["tf", "pkey", "time"])
def test_batch_mode(
    store: DiskgraphFeatureStore,
    safe: bool,
    attribute: str,
) -> None:
    """Shared batching preserves TensorFrames, key arrays, and temporal tensors."""
    expected = store["items", attribute, torch.arange(4)]
    context = store.safe_batch_mode if safe else store.batch_mode
    with (
        patch.object(store, "_get_tensor", wraps=store._get_tensor) as read,
        context(3),
    ):
        result = store["items", attribute, :]
    assert read.call_count == 2
    assert store._batch_size is None
    if isinstance(result, TensorFrame):
        assert result == expected
    elif isinstance(result, torch.Tensor):
        torch.testing.assert_close(result, expected)
    else:
        np.testing.assert_array_equal(result, expected)


def test_safe_batch_retry(store: DiskgraphFeatureStore) -> None:
    """Shared safe batching retries smaller requests and preserves row order."""
    read = store._get_tensor

    def fetch(attr: FeatureStoreAttr) -> FeatureStoreData | None:
        if store._batch_size == 4:
            raise RuntimeError("request too large")
        return read(attr)

    with (
        patch.object(store, "_get_tensor", side_effect=fetch),
        store.safe_batch_mode(batch_size=4, min_batch_size=2),
    ):
        result = store["items", "pkey", :]
        assert store._batch_size == 2
    np.testing.assert_array_equal(result, [0, 10, 20, 30])
    assert store._batch_size is None
    assert store._min_batch_size is None


def test_query_validation(store: DiskgraphFeatureStore) -> None:
    """Invalid table, attribute, and unbounded feature queries fail clearly."""
    with pytest.raises(KeyError, match="not fully specified"):
        store.get_tensor("items", "tf")
    with pytest.raises(KeyError, match="does not exist"):
        store["missing", "tf", torch.tensor([0])]
    with pytest.raises(KeyError, match="does not exist"):
        store["items", "y", torch.tensor([0])]
    with pytest.raises(KeyError, match="mini-batch"):
        store["items", "tf", :]
    with pytest.raises(KeyError, match="does not exist"):
        store["items", "df", torch.tensor([0])]
    with pytest.raises(NotImplementedError, match="DiskGraph runtime"):
        store.get_node_id("items", np.array([10]))


def test_multi_get_tensor(store: DiskgraphFeatureStore) -> None:
    """Multiple reads share the same requested order across feature attributes."""
    ids = torch.tensor([3, 1])
    keys, timestamps = store.multi_get_tensor(
        [
            FeatureStoreAttr("items", "pkey", ids),
            FeatureStoreAttr("items", "time", ids),
        ]
    )
    np.testing.assert_array_equal(keys, [30, 10])
    torch.testing.assert_close(timestamps, torch.tensor([30, 10]) * 1_000_000_000)
