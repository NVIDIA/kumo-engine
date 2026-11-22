# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MessagePack/NumPy graph parity, buffer ownership, and actor integration."""

from __future__ import annotations

import gc
from dataclasses import replace
from unittest.mock import Mock, patch

import msgpack
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from diskgraph.runtime_batch import RuntimeBatch
from kumo_core.diskgraph.batch_contract import (
    HeteroBatchPayload,
    SamplingContractError,
)
from kumo_core.diskgraph.hetero_wire import (
    pack_python_hetero,
    unpack_python_hetero,
)
from kumo_core.diskgraph.ray_sampling import RayRuntimeIndex, _SamplerReplica
from kumo_core.diskgraph.runtime_adapter import runtime_batch_to_hetero
from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumoapi.typing import Stype
from tests.distributed.test_ray_sampling import _batch, _config, _request
from torch_frame import TensorFrame
from torch_frame.data import MultiNestedTensor
from torch_geometric.data import HeteroData


def _graph() -> HeteroData:
    graph = HeteroData()
    node = graph["nodes"]
    node.n_id = torch.tensor([11, 12], dtype=torch.int64)
    node.batch = torch.tensor([0, 1], dtype=torch.int64)
    node.y = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    node.batch_size = 2
    node.num_sampled_nodes = [2, 0]
    node.pkey = np.empty(0, dtype=object)
    node.tf = TensorFrame(
        {
            Stype.numerical: torch.tensor([[1.5], [float("nan")]], dtype=torch.float32),
            Stype.categorical: torch.tensor([[1], [0]], dtype=torch.int64),
            Stype.timestamp: torch.arange(32, dtype=torch.float64).reshape(2, 1, 16),
            Stype.multicategorical: MultiNestedTensor(
                num_rows=2,
                num_cols=1,
                values=torch.tensor([1, 2], dtype=torch.int64),
                offset=torch.tensor([0, 2, 2], dtype=torch.int64),
            ),
        },
        {
            kind: [kind.value]
            for kind in (
                Stype.numerical,
                Stype.categorical,
                Stype.timestamp,
                Stype.multicategorical,
            )
        },
        num_rows=2,
    )
    edge = graph["nodes", "connects", "nodes"]
    edge.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    edge.num_sampled_edges = [2]
    return graph


def _assert_graph(actual: HeteroData, expected: HeteroData) -> None:
    assert actual.node_types == expected.node_types
    assert actual.edge_types == expected.edge_types
    for name in expected.node_types:
        one, two = actual[name], expected[name]
        assert one.batch_size == two.batch_size
        assert one.num_sampled_nodes == two.num_sampled_nodes
        for key in ("n_id", "batch", "y"):
            torch.testing.assert_close(one[key], two[key], equal_nan=True)
        assert one.tf.col_names_dict == two.tf.col_names_dict
        for kind, value in two.tf.feat_dict.items():
            if isinstance(value, MultiNestedTensor):
                torch.testing.assert_close(one.tf.feat_dict[kind].values, value.values)
                torch.testing.assert_close(one.tf.feat_dict[kind].offset, value.offset)
            else:
                torch.testing.assert_close(
                    one.tf.feat_dict[kind], value, equal_nan=True
                )
    for key in expected.edge_types:
        torch.testing.assert_close(actual[key].edge_index, expected[key].edge_index)
        assert actual[key].num_sampled_edges == expected[key].num_sampled_edges


def test_wire_preserves_graph_values_and_buffer_ownership() -> None:
    """Metadata stays separate and decoded tensors retain the numeric buffers."""
    graph = _graph()
    metadata, arrays = pack_python_hetero(graph)
    assert isinstance(metadata, bytes)
    assert msgpack.unpackb(metadata, raw=False)["version"] == 1
    assert any(np.shares_memory(array, graph["nodes"].n_id.numpy()) for array in arrays)
    decoded = unpack_python_hetero(metadata, arrays)
    _assert_graph(decoded, graph)
    assert np.shares_memory(decoded["nodes"].n_id.numpy(), graph["nodes"].n_id.numpy())
    del graph, arrays, metadata
    gc.collect()
    assert decoded["nodes"].n_id.tolist() == [11, 12]


def test_wire_copies_strided_views_only() -> None:
    """Noncontiguous tensors are packed contiguously without changing values."""
    graph = _graph()
    graph["nodes"].n_id = torch.arange(4, dtype=torch.int64)[::2]
    metadata, arrays = pack_python_hetero(graph)
    assert not any(
        np.shares_memory(array, graph["nodes"].n_id.numpy()) for array in arrays
    )
    _assert_graph(unpack_python_hetero(metadata, arrays), graph)


@pytest.mark.parametrize(
    "failure", ["version", "dtype", "shape", "duplicate", "unused", "offset", "edge"]
)
def test_wire_rejects_invalid_layout(failure: str) -> None:
    """Invalid descriptors cannot silently change tensor layout or graph topology."""
    metadata, arrays = pack_python_hetero(_graph())
    spec = msgpack.unpackb(metadata, raw=False)
    node = spec["nodes"]["nodes"]
    attribute = node["attributes"]["n_id"]
    if failure == "version":
        spec["version"] = 2
    elif failure == "dtype":
        attribute["dtype"] = "float32"
    elif failure == "shape":
        attribute["shape"] = [999]
    elif failure == "duplicate":
        node["attributes"]["duplicate"] = attribute
    elif failure == "unused":
        arrays.append(np.empty(0, dtype=np.int64))
    elif failure == "offset":
        nested = next(f for f in node["features"] if f["kind"] == "multicategorical")
        arrays[nested["offset"]["buffer"]] = np.array([0, 9, 2], dtype=np.int64)
    else:
        spec["edges"][0][0][2] = "missing"
    with pytest.raises((ValueError, TypeError)):
        unpack_python_hetero(msgpack.packb(spec, use_bin_type=True), arrays)


def _encode_runtime(batch: RuntimeBatch) -> HeteroData:
    return runtime_batch_to_hetero(
        batch,
        train_table_name="nodes",
        seed_node_type="nodes",
        attach_pkeys=False,
        mapper_bank={"nodes": {"time": TimestampTensorMapper()}},
        feature_schema={"node_types": {"nodes": {"mapper_key": "nodes"}}},
    )[0]


def test_actor_uses_existing_python_encoder_and_emits_separate_buffers() -> None:
    """Actor responses contain NumPy arrays rather than pickled tensor objects."""
    request = _request()
    replica = _SamplerReplica.__new__(_SamplerReplica)
    replica.config = replace(_config(), batch_encoder=_encode_runtime)
    replica.index = Mock()
    replica.contexts = []
    raw = _batch(request)
    context = Mock()
    context.sample_request.return_value = raw
    replica.index.iter.return_value = (context, 2)
    response = replica.sample(request)
    assert isinstance(response, HeteroBatchPayload)
    assert all(isinstance(array, np.ndarray) for array in response.buffers)
    decoded = response.decode()
    expected = _encode_runtime(raw)
    torch.testing.assert_close(
        decoded["nodes"].tf.feat_dict[Stype.timestamp],
        expected["nodes"].tf.feat_dict[Stype.timestamp],
    )
    torch.testing.assert_close(decoded["nodes"].n_id, expected["nodes"].n_id)
    assert not hasattr(response, "request")


def test_reference_client_requires_encoded_response() -> None:
    """The trainer consumes encoded graphs and rejects legacy raw responses."""
    request = _request()
    metadata, arrays = pack_python_hetero(_graph())
    encoded = HeteroBatchPayload(metadata, arrays, 0.1)
    client = RayRuntimeIndex.__new__(RayRuntimeIndex)
    with patch.object(client, "_receive", return_value=encoded):
        _assert_graph(client._sample(request), _graph())
    with (
        patch.object(client, "_receive", return_value=_batch(request)),
        pytest.raises(SamplingContractError, match="encoded graph"),
    ):
        client._sample(request)


def test_wire_rejects_extra_edge_attributes() -> None:
    """Unsupported edge data must fail instead of suggesting another serializer."""
    graph = _graph()
    graph["nodes", "connects", "nodes"].score = torch.ones(2)
    with pytest.raises(NotImplementedError, match="unsupported"):
        pack_python_hetero(graph)
