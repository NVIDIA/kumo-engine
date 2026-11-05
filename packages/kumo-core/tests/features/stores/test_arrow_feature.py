# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Arrow-backed in-memory feature storage."""

from types import SimpleNamespace
from unittest.mock import mock_open, patch

import numpy as np
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper, TensorMapper
from kumo_core.features.stores.feature import FeatureStoreTableSchema
from kumo_core.features.stores.in_memory_feature import (
    InMemoryFeatureStore,
    _as_table,
    _as_timestamp,
    _take_rows,
    _timestamp_tensor,
)
from kumoapi.encoder import Numerical
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame


@pytest.mark.parametrize("num_rows", [0, 3])
@pytest.mark.parametrize("with_features", [False, True])
@pytest.mark.parametrize("pmm", [False, True])
@pytest.mark.parametrize("as_batch", [False, True])
def test_arrow_storage(
    num_rows: int,
    with_features: bool,
    pmm: bool,
    as_batch: bool,
) -> None:
    """Arrow inputs preserve row counts in eager and on-demand storage modes."""
    batch = pa.record_batch({"value": range(num_rows)})
    table = batch if as_batch else pa.Table.from_batches([batch])
    mappers: dict[str, TensorMapper] = (
        {"value": NumericalTensorMapper()} if with_features else {}
    )
    store = InMemoryFeatureStore(
        {"table": table},
        {"table": FeatureStoreTableSchema(mapper_dict=mappers)},
        pmm=pmm,
    )

    result = store._get_tf("table", None)

    assert result.num_rows == num_rows
    if not pmm:
        assert store._raw_tables["table"].equals(_as_table(table))
    assert "df" not in store._data["table"]
    if with_features:
        assert result.col_names_dict == {Stype.numerical: ["value"]}
        features = result.feat_dict[Stype.numerical]
        assert isinstance(features, Tensor)
        assert features.equal(
            torch.arange(num_rows, dtype=torch.get_default_dtype()).view(-1, 1)
        )
    else:
        assert result.feat_dict == {}


@pytest.mark.parametrize(
    "index,expected",
    [
        (None, [0, 1, 2, 3]),
        (slice(1, 3), [1, 2]),
        (slice(None, None, -1), [3, 2, 1, 0]),
        (slice(3, 1), []),
        ([3, 1, 3], [3, 1, 3]),
        ([-1, 0], [3, 0]),
        ([], []),
        (np.array([True, False, True, False]), [0, 2]),
        (torch.tensor([2, 0]), [2, 0]),
        (1, [1]),
    ],
)
def test_take_rows(
    index: Tensor | slice | np.ndarray | list[int] | int | None,
    expected: list[int],
) -> None:
    """Positional selections preserve duplicates, order, and zero-column rows."""
    table = pa.table({"value": [0, 1, 2, 3]})

    result = _take_rows(table, index)

    assert result.column("value").to_pylist() == expected
    assert _take_rows(table.select([]), index).num_rows == len(expected)


def test_as_table_shares_buffers() -> None:
    """Wrapping a batch retains its buffers and existing tables are reused."""
    batch = pa.record_batch({"value": [1, 2]})

    table = _as_table(batch)

    assert table.column(0).chunk(0).buffers() == batch.column(0).buffers()
    assert _as_table(table) is table


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timestamp_units(unit: str) -> None:
    """Timestamp units normalize to nanoseconds and preserve missing sentinels."""
    values = pa.array([1, None], type=pa.timestamp(unit))
    expected = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[unit]

    normalized = _as_timestamp(values)
    result = _timestamp_tensor(values, -1)

    assert normalized.type == pa.timestamp("ns")
    assert result.tolist() == [expected, -1]
    assert _timestamp_tensor(values).tolist() == [expected, np.iinfo(np.int64).min]


def test_timestamp_strings_and_empty_values() -> None:
    """Year strings keep legacy parsing; missing and empty inputs remain valid."""
    assert _timestamp_tensor(pa.array(["1970", "", None]), -1).tolist() == [0, -1, -1]
    assert _timestamp_tensor(pa.nulls(0)).numel() == 0


def test_sanitize_arrow_order_and_nan() -> None:
    """Deduplication precedes time filtering and drops both null and NaN keys."""
    table = pa.table(
        {
            "id": [3.0, 1.0, 3.0, float("nan"), None, 2.0],
            "time": [None, 10.0, 20.0, 30.0, 40.0, float("nan")],
        }
    )

    result = InMemoryFeatureStore.sanitize(
        {"table": table}, {"table": "id"}, {"table": "time"}
    )

    assert result["table"].to_pydict() == {"id": [1.0], "time": [10.0]}
    assert table.num_rows == 6


def test_primary_key_lookup() -> None:
    """Arrow string primary keys support ordered repeated lookups and errors."""
    store = InMemoryFeatureStore(
        {"table": pa.table({"id": ["c", "a", "b"]})},
        {"table": FeatureStoreTableSchema(mapper_dict={}, pkey_name="id")},
    )

    result = store.get_node_id("table", np.array(["b", "c", "b"]))

    assert result.tolist() == [2, 0, 2]
    with pytest.raises(KeyError, match="missing"):
        store.get_node_id("table", np.array(["missing"]))


@pytest.mark.parametrize("dtype", [pa.int32(), pa.int64(), pa.date32()])
def test_timestamp_numeric_and_date_types(dtype: pa.DataType) -> None:
    """Arrow date and integer timestamp columns normalize without pandas."""
    values = pa.chunked_array([[1], [None]], type=dtype)
    expected = 86_400 * 10**9 if pa.types.is_date(dtype) else 1

    result = _timestamp_tensor(values, -1)

    assert result.tolist() == [expected, -1]


def test_diskgraph_arrow_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """DiskGraph initializes empty Arrow features using schema and node metadata."""
    from kumo_core.diskgraph import stores

    schema = SimpleNamespace(table_names=["items"], pkey_dict={"items": "id"})
    monkeypatch.setattr(stores, "load_diskgraph_schema", lambda _: schema)
    metadata = '{"node_types": [{"name": "items", "max_id": 9}]}'
    encoders = EncodersConfig()
    encoders["items"] = {"value": Numerical()}
    with patch("builtins.open", mock_open(read_data=metadata)):
        store = stores.DiskgraphFeatureStore.for_explain(
            index_path="/index",
            encoders=encoders,
            entity_table_names=["items"],
            time_col_dict={"items": "time"},
        )

    features = store._data["items"]["tf"]
    assert isinstance(features, TensorFrame)
    assert features.num_rows == 0
    assert store._get_fused_tf_empty("items").num_rows == 0
    assert store._diskgraph_num_nodes == {"items": 10}
    assert store.entity_table_names == ["items"]


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "keys",
    [
        ["b", None, "a", "a", None, "c", "b"],
        [20, None, 10, 10, None, 30, 20],
    ],
)
def test_sanitize_dictionary_primary_keys(
    keys: list[str | int | None],
    chunked: bool,
) -> None:
    """Deduplicate dictionary values across chunks while discarding null keys."""
    encoded = pa.array(keys).dictionary_encode()
    if chunked:
        encoded = pa.chunked_array(
            [
                pa.array(keys[:3]).dictionary_encode(),
                pa.array(keys[3:]).dictionary_encode(),
            ]
        )
    table = pa.table({"id": encoded, "row": range(len(keys))})

    result = InMemoryFeatureStore.sanitize({"table": table}, {"table": "id"}, {})

    assert result["table"].to_pydict() == {
        "id": [keys[0], keys[2], keys[5]],
        "row": [0, 2, 5],
    }
    assert result["table"].schema == table.schema
    assert table.column("id").to_pylist() == keys
