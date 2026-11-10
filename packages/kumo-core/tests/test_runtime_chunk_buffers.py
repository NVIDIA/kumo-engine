# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native feature batches retain their Arrow buffers through tensor encoding."""

from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
from kumo_core.diskgraph.runtime_adapter import (
    _record_batches_to_table,
    _values_to_tensor,
    attach_tensor_frames,
    engine_int64_to_ns,
)
from kumo_core.diskgraph.stores import _iter_node_column
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper
from kumoapi.typing import Stype
from torch import Tensor
from torch_geometric.data import HeteroData


@pytest.mark.parametrize("empty", [False, True])
def test_feature_batches_preserve_original_buffers(empty: bool) -> None:
    """Runtime feature collection keeps source chunks rather than copying a table."""
    column = pa.array([] if empty else [1.0, None, 3.0])
    batches = [
        pa.record_batch({"feature": column.slice(0, 1)}),
        pa.record_batch({"feature": column.slice(1)}),
    ]

    table = _record_batches_to_table(batches)

    assert isinstance(table, pa.Table)
    assert table.num_rows == len(column)
    assert table.column("feature").num_chunks == 2
    for chunk, batch in zip(table.column("feature").chunks, batches, strict=True):
        original = batch.column("feature")
        assert chunk.offset == original.offset
        assert [buffer.address if buffer else None for buffer in chunk.buffers()] == [
            buffer.address if buffer else None for buffer in original.buffers()
        ]


def test_single_feature_batch_is_wrapped_without_copying() -> None:
    """A lone RecordBatch uses the same table representation and source buffer."""
    batch = pa.record_batch({"feature": [1.0]})

    table = _record_batches_to_table([batch])

    assert isinstance(table, pa.Table)
    assert table.num_rows == 1
    column = table.column("feature")
    assert column.num_chunks == 1
    assert column.chunk(0).buffers()[1].address == batch.column(0).buffers()[1].address


def test_runtime_encodes_original_feature_chunks() -> None:
    """The runtime forwards each batch's original array to feature encoding."""
    batches = [
        pa.record_batch({"feature": [1.0, None]}),
        pa.record_batch({"feature": [3.0]}),
    ]
    mapper = NumericalTensorMapper()
    hetero = HeteroData()
    table = _record_batches_to_table(batches)

    with patch.object(mapper, "forward", wraps=mapper.forward) as forward:
        attach_tensor_frames(
            hetero,
            {"nodes": table},
            {"node_mapper": {"feature": mapper}},
            {"node_types": {"nodes": {"mapper_key": "node_mapper"}}},
        )

    encoded = hetero["nodes"].tf.feat_dict[Stype.numerical]
    assert isinstance(encoded, Tensor)
    torch.testing.assert_close(
        encoded,
        torch.tensor([[1.0], [float("nan")], [3.0]]),
        equal_nan=True,
    )
    assert [len(call.args[0]) for call in forward.call_args_list] == [2, 1]


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.long])
@pytest.mark.parametrize("nullable", [False, True])
def test_runtime_scalar_chunks(dtype: torch.dtype | None, nullable: bool) -> None:
    """Scalar destination allocation preserves dtype promotion and row order."""
    array = pa.array([1, 2, None if nullable else 3], type=pa.int64())
    chunks = pa.chunked_array([array.slice(0, 0), array.slice(0, 2), array.slice(2)])
    expected = _values_to_tensor(array, dtype=dtype)

    actual = _values_to_tensor(chunks, dtype=dtype)

    torch.testing.assert_close(actual, expected, equal_nan=True)
    actual.fill_(0)
    assert chunks.to_pylist() == array.to_pylist()


def test_runtime_empty_scalar_chunks() -> None:
    """No-chunk scalar inputs retain the requested destination dtype."""
    result = _values_to_tensor(pa.chunked_array([], pa.int64()), dtype=torch.long)

    assert result.shape == (0,)
    assert result.dtype == torch.long


def test_runtime_chunked_timestamp_scaling() -> None:
    """Timestamp scaling handles chunked ticks without combining the input."""
    ticks = pa.chunked_array([pa.array([2, None]), pa.array([-1])])

    assert engine_int64_to_ns(ticks, "ns") is ticks
    assert engine_int64_to_ns(ticks, "ms").to_pylist() == [2_000_000, None, -1_000_000]


def test_node_column_scan_preserves_response_chunks() -> None:
    """Node column scans expose each response buffer with its original row range."""
    batches = [
        pa.record_batch({"feature": [1, 2]}),
        pa.record_batch({"feature": pa.array([], type=pa.int64())}),
        pa.record_batch({"feature": [3]}),
    ]
    engine = Mock()
    engine.fetch_features.return_value = [
        {"type": "feature", "batch": batch} for batch in batches
    ]

    chunks = list(_iter_node_column(engine, "nodes", "feature", 10, 13))

    assert [row for row, _ in chunks] == [10, 12]
    for (_, chunk), batch in zip(chunks, [batches[0], batches[2]], strict=True):
        assert (
            chunk.buffers()[1].address == batch.column("feature").buffers()[1].address
        )


def test_rhs_seed_times_encode_chunks() -> None:
    """RHS collation scales chunked seed timestamps into the destination tensor."""
    hetero = HeteroData()
    hetero["rhs"].num_nodes = 3
    module = Mock(index_np_time_unit="ms")
    module._to_hetero.return_value = hetero
    batch = {
        "task": {"seed_timestamps": pa.chunked_array([pa.array([1, 2]), pa.array([3])])}
    }

    result = DiskgraphLightningDataModule._rhs_to_hetero(module, batch, "rhs")

    assert result is hetero
    assert result["rhs"].seed_time.tolist() == [1_000_000, 2_000_000, 3_000_000]
