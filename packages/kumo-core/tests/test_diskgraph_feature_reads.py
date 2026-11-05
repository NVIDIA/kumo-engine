# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np
import orjson
import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.stores import DiskgraphFeatureStore
from kumo_core.encoders.config import EncodersConfig
from kumo_core.modeling.retrieval.rhs_embedding import RHSEmbedding
from kumoapi.encoder import NAStrategy, Numerical
from kumoapi.model_plan import RHSEmbeddingMode
from torch_frame import TensorFrame


@pytest.fixture(autouse=True)
def mock_native_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the native module; these tests exercise the Python store boundary."""
    monkeypatch.setitem(sys.modules, "diskgraph", SimpleNamespace(SamplerEngine=Mock()))


@pytest.fixture
def store() -> DiskgraphFeatureStore:
    """Return a metadata-only store with mocked snapshot files."""
    schema = SimpleNamespace(table_names=["items"], pkey_dict={"items": "id"})
    metadata = '{"node_types": [{"name": "items", "max_id": 5}]}'
    with (
        patch("kumo_core.diskgraph.stores.load_diskgraph_schema", return_value=schema),
        patch("builtins.open", mock_open(read_data=metadata)),
    ):
        return DiskgraphFeatureStore.for_explain(
            index_path="/snapshot",
            encoders=EncodersConfig({"items": {"value": Numerical()}}),
            entity_table_names=["items"],
            time_col_dict={"items": "created_at"},
            end_time_col_dict={"items": "ended_at"},
        )


def test_metadata_and_lazy_engine(store: DiskgraphFeatureStore) -> None:
    """Construction loads metadata without creating a native engine."""
    assert store._engine is None
    assert store.table_names == ["items"]
    assert store.entity_table_names == ["items"]
    assert store.train_table_name is None
    assert store.get_tensor_size("items", "tf") == (6,)
    assert store.get_pkey_name("items") == "id"
    assert store.get_time_col_name("items") == "created_at"
    assert store.get_end_time_col_name("items") == "ended_at"


@pytest.mark.parametrize("max_id", [None, -1])
def test_empty_table_metadata(max_id: int | None) -> None:
    """Empty tables keep their size without an in-memory fallback."""
    schema = SimpleNamespace(table_names=["items"], pkey_dict={"items": "id"})
    metadata = orjson.dumps({"node_types": [{"name": "items", "max_id": max_id}]})
    with (
        patch("kumo_core.diskgraph.stores.load_diskgraph_schema", return_value=schema),
        patch("builtins.open", mock_open(read_data=metadata)),
    ):
        empty_store = DiskgraphFeatureStore.for_explain(
            index_path="/snapshot",
            encoders=EncodersConfig(),
            entity_table_names=["items"],
        )
    assert empty_store.get_tensor_size("items", "tf") == (0,)
    assert len(empty_store["items", "pkey", :]) == 0


def test_primary_key_order_and_engine_reuse(store: DiskgraphFeatureStore) -> None:
    """Indexed lookups preserve request order and share one lazy engine."""
    batch = pa.record_batch({"id": ["item-4", "item-1", "item-4"]})
    engine = Mock()
    engine.fetch_features.return_value = [{"type": "feature", "batch": batch}]
    with patch("diskgraph.SamplerEngine", return_value=engine) as constructor:
        for _ in range(2):
            result = store["items", "pkey", torch.tensor([4, 1, 4])]
            np.testing.assert_array_equal(result, ["item-4", "item-1", "item-4"])
    constructor.assert_called_once_with("/snapshot")
    engine.fetch_features.assert_called_with(
        [
            {
                "node_type": "items",
                "node_ids": [4, 1, 4],
                "feature_names": ["id"],
            }
        ]
    )


def test_tensor_frame_read(store: DiskgraphFeatureStore) -> None:
    """Encoded features retain the order and row count of the requested nodes."""
    engine = Mock()
    engine.fetch_features.return_value = [
        {
            "type": "feature",
            "batch": pa.record_batch({"value": [4.0, 1.0]}),
        }
    ]
    with patch("diskgraph.SamplerEngine", return_value=engine):
        result = store["items", "tf", torch.tensor([4, 1])]
    assert isinstance(result, TensorFrame)
    assert result.num_rows == 2
    values = next(iter(result.feat_dict.values()))
    torch.testing.assert_close(values.flatten(), torch.tensor([4.0, 1.0]))


@pytest.mark.parametrize(
    "mode",
    [
        RHSEmbeddingMode.FEATURE,
        RHSEmbeddingMode.SHALLOW_FEATURE,
        RHSEmbeddingMode.FUSION,
    ],
)
@pytest.mark.parametrize("filtered", [False, True])
def test_rhs_feature_prefetch(
    store: DiskgraphFeatureStore,
    mode: RHSEmbeddingMode,
    filtered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RHS modes fetch and cache indexed DiskGraph features across chunks."""
    monkeypatch.setattr("kumo_core.modeling.retrieval.rhs_embedding.FS_BATCH_SIZE", 1)
    model = RHSEmbedding(
        emb_mode=mode,
        num_nodes=6,
        embedding_dim=4,
        encoder_dict={"value": Numerical(na_strategy=NAStrategy.ZERO)},
    )
    node_ids = [4, 1] if filtered else list(range(6))
    if filtered:
        model.filter(torch.tensor(node_ids))
    engine = Mock()
    engine.fetch_features.side_effect = [
        [{"type": "feature", "batch": pa.record_batch({"value": [float(node)]})}]
        for node in node_ids
    ]
    with patch("diskgraph.SamplerEngine", return_value=engine):
        model.fetch_feats(store, "items")
        if not filtered:
            model.fetch_feats(store, "items")

    assert model._feats is not None
    assert all(features.size(0) == len(node_ids) for features in model._feats)
    assert model._filtered_feats is filtered
    assert [
        call.args[0][0]["node_ids"] for call in engine.fetch_features.call_args_list
    ] == [[node] for node in node_ids]


@pytest.mark.parametrize("attribute", ["time", "end_time"])
def test_temporal_read_nanoseconds(
    store: DiskgraphFeatureStore,
    attribute: str,
) -> None:
    """Second-resolution timestamps are normalized to nanoseconds."""
    column = "created_at" if attribute == "time" else "ended_at"
    engine = Mock()
    engine.fetch_features.return_value = [
        {
            "type": "feature",
            "batch": pa.record_batch({column: pa.array([123, 456], pa.timestamp("s"))}),
        }
    ]
    with patch("diskgraph.SamplerEngine", return_value=engine):
        result = store["items", attribute, torch.tensor([4, 1])]
    torch.testing.assert_close(result, torch.tensor([123, 456]) * 1_000_000_000)


def test_full_primary_keys_are_chunked(store: DiskgraphFeatureStore) -> None:
    """Full-table reads preserve keys across engine request boundaries."""
    store._diskgraph_num_nodes["items"] = 50_001
    engine = Mock()
    engine.fetch_features.side_effect = [
        [{"type": "feature", "batch": pa.record_batch({"id": list(range(50_000))})}],
        [{"type": "feature", "batch": pa.record_batch({"id": [50_000]})}],
    ]
    with patch("diskgraph.SamplerEngine", return_value=engine):
        result = store["items", "pkey", :]
    np.testing.assert_array_equal(result, np.arange(50_001))
    requests = engine.fetch_features.call_args_list
    assert len(requests) == 2
    assert len(requests[0].args[0][0]["node_ids"]) == 50_000
    assert requests[1].args[0][0]["node_ids"] == [50_000]


@pytest.mark.parametrize("attribute", ["tf", "pkey", "time", "end_time"])
def test_empty_read_skips_engine(
    store: DiskgraphFeatureStore,
    attribute: str,
) -> None:
    """Empty node selections keep their schema without invoking DiskGraph."""
    with patch("diskgraph.SamplerEngine") as constructor:
        result = store["items", attribute, torch.empty(0, dtype=torch.long)]
    assert len(result) == 0
    constructor.assert_not_called()


def test_unencoded_table_preserves_row_count(store: DiskgraphFeatureStore) -> None:
    """A table without model features still represents every requested node."""
    store._table_schema_dict["items"].mapper_dict.clear()
    with patch("diskgraph.SamplerEngine") as constructor:
        result = store["items", "tf", torch.tensor([4, 1])]
    assert isinstance(result, TensorFrame)
    assert result.num_rows == 2
    assert result.feat_dict == {}
    constructor.assert_not_called()


def test_pickle_drops_native_engine(store: DiskgraphFeatureStore) -> None:
    """Serialization retains metadata and discards unpicklable native state."""
    store._engine = Mock()
    restored = pickle.loads(pickle.dumps(store))
    assert restored._engine is None
    assert restored.entity_table_names == ["items"]
    assert restored.get_tensor_size("items", "tf") == (6,)
    assert store._engine is not None


def test_missing_feature_events_fail(store: DiskgraphFeatureStore) -> None:
    """Missing rows must not produce fabricated primary keys."""
    engine = Mock()
    engine.fetch_features.return_value = []
    with (
        patch("diskgraph.SamplerEngine", return_value=engine),
        pytest.raises(RuntimeError, match="returned no features"),
    ):
        store["items", "pkey", torch.tensor([4, 1])]
