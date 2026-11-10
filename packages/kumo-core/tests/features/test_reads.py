# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for explicit, ordered native feature reads."""

import pickle
from decimal import Decimal
from unittest.mock import Mock, mock_open

import numpy as np
import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.metadata import DiskgraphSchema, load_diskgraph_schema
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    NumericalTensorMapper,
)
from kumo_core.features import reads
from kumo_core.features.stores.feature import FeatureStoreTableSchema
from kumo_core.modeling.retrieval.rhs_embedding import RHSEmbedding
from kumoapi.encoder import NAStrategy, Numerical
from kumoapi.typing import Stype


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> reads.DiskgraphFeatureContext:
    """Provide fitted metadata without accessing the filesystem or native code."""
    monkeypatch.setattr(
        reads, "_decimal_primary_key_arrow_type", Mock(return_value=None)
    )
    return reads.DiskgraphFeatureContext(
        index_path="/index",
        schema=DiskgraphSchema(
            table_names=["events"],
            pkey_dict={"events": "id"},
            fk_specs=[],
            raw_edge_types=[],
            edge_types=[],
            edge_type_remap={},
        ),
        table_schema_dict={
            "events": FeatureStoreTableSchema(
                mapper_dict={"value": NumericalTensorMapper()},
                time_col_name="created_at",
                end_time_col_name="ended_at",
            )
        },
        entity_table_names=["events"],
    )


def test_tensor_reads_preserve_duplicate_order_and_bound_requests(
    context: reads.DiskgraphFeatureContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk only requested IDs and retain the engine's response row order."""
    monkeypatch.setattr(reads, "READ_BATCH_SIZE", 2)
    engine = Mock()
    engine.fetch_features.side_effect = [
        [{"type": "feature", "batch": pa.record_batch({"value": [30.0, 10.0]})}],
        [{"type": "feature", "batch": pa.record_batch({"value": [30.0]})}],
    ]
    context._engine = engine

    frame = reads.read_tensor_frame(context, "events", torch.tensor([3, 1, 3]))

    assert frame.feat_dict[Stype.numerical].flatten().tolist() == [30.0, 10.0, 30.0]
    assert [
        call.args[0][0]["node_ids"] for call in engine.fetch_features.call_args_list
    ] == [[3, 1], [3]]
    assert frame.col_names_dict[Stype.numerical] == ["value"]


def test_empty_tensor_reads_preserve_schema_without_opening_engine(
    context: reads.DiskgraphFeatureContext,
) -> None:
    """An empty request retains all fitted feature columns."""
    context.table_schema_dict["events"].mapper_dict["category"] = (
        CategoricalTensorMapper([["a", 2], ["b", 1]])
    )

    frame = reads.read_tensor_frame(context, "events", torch.empty(0, dtype=torch.long))

    assert frame.num_rows == 0
    assert frame.col_names_dict[Stype.numerical] == ["value"]
    assert frame.col_names_dict[Stype.categorical] == ["category"]
    assert context._engine is None


def test_featureless_nodes_keep_requested_row_count(
    context: reads.DiskgraphFeatureContext,
) -> None:
    """Featureless tables still return one TensorFrame row per requested ID."""
    context.table_schema_dict["events"].mapper_dict.clear()

    frame = reads.read_tensor_frame(context, "events", torch.tensor([4, 2]))

    assert frame.num_rows == 2
    assert context._engine is None


@pytest.mark.parametrize("keys", [["c", "a", "c"], [300, 100, 300]])
def test_primary_keys_keep_native_values_and_duplicate_order(
    context: reads.DiskgraphFeatureContext,
    keys: list[str] | list[int],
) -> None:
    """Return real primary keys rather than positional or physical IDs."""
    engine = Mock()
    engine.fetch_features.return_value = [
        {"type": "feature", "batch": pa.record_batch({"id": keys})}
    ]
    context._engine = engine

    result = reads.read_primary_keys(context, "events", torch.tensor([3, 1, 3]))

    assert result.tolist() == keys
    assert result.flags.writeable


def test_decimal_primary_keys_keep_their_declared_type(
    context: reads.DiskgraphFeatureContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore decimal types from the index's feature-fetch string view."""
    monkeypatch.setattr(
        reads,
        "_decimal_primary_key_arrow_type",
        Mock(return_value=pa.decimal128(10, 2)),
    )
    context._engine = Mock()
    context._engine.fetch_features.return_value = [
        {"type": "feature", "batch": pa.record_batch({"id": ["123.45"]})}
    ]

    result = reads.read_primary_keys(context, "events", torch.tensor([3]))

    assert result.tolist() == [Decimal("123.45")]


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_node_times_normalize_resolution_and_preserve_nulls(
    context: reads.DiskgraphFeatureContext,
    unit: str,
) -> None:
    """Selection receives nanoseconds regardless of the engine's clock unit."""
    context._engine = Mock()
    context._engine.fetch_features.return_value = [
        {
            "type": "feature",
            "batch": pa.record_batch(
                {"created_at": pa.array([1, None], type=pa.timestamp(unit))}
            ),
        }
    ]

    result = reads.read_node_times(context, "events", torch.tensor([3, 1]))

    factor = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}[unit]
    assert result.tolist() == [factor, np.iinfo(np.int64).min]


def test_missing_columns_and_incomplete_reads_raise(
    context: reads.DiskgraphFeatureContext,
) -> None:
    """Missing data must not silently become fabricated primary keys."""
    context._engine = Mock()
    context._engine.fetch_features.return_value = []
    with pytest.raises(RuntimeError, match="returned 0 rows"):
        reads.read_primary_keys(context, "events", torch.tensor([3]))

    context._engine.fetch_features.return_value = [
        {"type": "feature", "batch": pa.record_batch({"other": [123]})}
    ]
    with pytest.raises(KeyError, match="omitted columns"):
        reads.read_primary_keys(context, "events", torch.tensor([3]))


def test_context_reopens_native_handle_after_pickling(
    context: reads.DiskgraphFeatureContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers receive metadata and open their own unpicklable native handle."""
    engine_factory = Mock(side_effect=[Mock(), Mock()])
    monkeypatch.setattr(reads, "_node_column_engine", engine_factory)
    original_engine = context.engine

    restored = pickle.loads(pickle.dumps(context))

    assert restored._engine is None
    assert restored.engine is not original_engine
    assert engine_factory.call_count == 2
    assert restored.table_names == ["events"]
    assert restored.get_pkey_name("events") == "id"
    assert restored.get_time_col_name("events") == "created_at"
    assert restored.time_col_dict == {"events": "created_at"}
    assert restored.train_table_name is None
    assert restored.get_tensor_mapper("events", "created_at").stype == Stype.timestamp
    assert restored.get_tensor_mapper("events", "ended_at").stype == Stype.timestamp
    with pytest.raises(KeyError, match="Unknown column"):
        restored.get_tensor_mapper("events", "missing")


def test_schema_sizes_use_physical_node_id_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep sparse ID ranges and explicitly empty tables in schema metadata."""
    metadata = (
        b'{"node_types": [{"name": "events", "max_id": 7}, '
        b'{"name": "empty", "max_id": -1}]}'
    )
    monkeypatch.setattr("builtins.open", mock_open(read_data=metadata))

    schema = load_diskgraph_schema("/index")

    assert schema.num_nodes_dict == {"events": 8, "empty": 0}


def test_rhs_embedding_native_reads_match_prefetched_features(
    context: reads.DiskgraphFeatureContext,
) -> None:
    """Fusion produces the same embeddings with direct and prefetched reads."""
    context._engine = Mock()
    context._engine.fetch_features.return_value = [
        {"type": "feature", "batch": pa.record_batch({"value": [10.0, 30.0]})}
    ]
    model = RHSEmbedding(
        emb_mode="fusion",
        num_nodes=5,
        embedding_dim=4,
        encoder_dict={"value": Numerical(scaler=None, na_strategy=NAStrategy.ZERO)},
    )
    node_ids = torch.tensor([1, 3])
    model.filter(node_ids)

    with torch.no_grad():
        direct = model(node_ids, context, table_name="events")
    model.fetch_feats(context, table_name="events")
    prefetched = model(node_ids)

    assert torch.allclose(direct, prefetched)
    assert context._engine.fetch_features.call_count == 2
