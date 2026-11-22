# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MessagePack graph metadata with out-of-band NumPy tensor buffers."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import HeteroData

TensorBuffer = NDArray[np.int64 | np.float32 | np.float64]


class _TensorSpec(TypedDict):
    """Tensor descriptor in the versioned MessagePack wire format."""

    buffer: int
    dtype: str
    shape: list[int]


def pack_python_hetero(data: HeteroData) -> tuple[bytes, list[TensorBuffer]]:
    """Export Python-assembled CPU tensors using the version-1 wire.

    Args:
        data: HeteroData produced by the shared Python runtime adapter.

    Returns:
        MessagePack metadata and contiguous NumPy views retaining tensor owners.
        Only noncontiguous tensors require a buffer copy.

    Raises:
        NotImplementedError: If attributes or features exceed the wire format.
        ValueError: If tensors are not CPU int64/float32/float64 arrays.
    """
    import msgpack
    from torch_frame.data import MultiNestedTensor

    buffers: list[TensorBuffer] = []

    def describe(tensor: torch.Tensor) -> _TensorSpec:
        if tensor.device.type != "cpu" or tensor.dtype not in (
            torch.int64,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Wire tensors must be CPU int64/float32/float64")
        array = tensor.detach().contiguous().numpy()
        spec: _TensorSpec = {
            "buffer": len(buffers),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        buffers.append(array.reshape(-1))
        return spec

    if data._global_store:
        raise NotImplementedError(
            "Global attributes are unsupported by this wire format"
        )
    nodes = {}
    batch_size = None
    for name in data.node_types:
        block = data[name]
        if batch_size is not None and block.batch_size != batch_size:
            raise ValueError("All node stores must have the same batch_size")
        batch_size = block.batch_size
        attributes = {}
        for key, value in block.items():
            if key in ("tf", "batch_size", "num_sampled_nodes", "pkey"):
                continue
            if not isinstance(value, torch.Tensor):
                raise NotImplementedError(f"Unsupported node attribute {name}.{key}")
            attributes[key] = describe(value)
        if "pkey" in block and len(block.pkey):
            raise NotImplementedError(
                "Primary keys are unsupported by this wire format"
            )
        features = []
        for kind, tensor in block.tf.feat_dict.items():
            if kind.value not in (
                "numerical",
                "categorical",
                "timestamp",
                "multicategorical",
            ):
                raise NotImplementedError(
                    f"Feature {kind} is unsupported by this wire format"
                )
            nested = isinstance(tensor, MultiNestedTensor)
            features.append(
                {
                    "kind": kind.value,
                    "columns": block.tf.col_names_dict[kind],
                    "value": describe(tensor.values if nested else tensor),
                    "offset": describe(tensor.offset) if nested else None,
                }
            )
        nodes[name] = {
            "attributes": attributes,
            "features": features,
            "rows": block.tf.num_rows,
            "counts": block.get("num_sampled_nodes"),
            "empty_pkey": "pkey" in block,
        }
    edges = []
    for key in data.edge_types:
        block = data[key]
        if set(block) - {"edge_index", "num_sampled_edges"}:
            raise NotImplementedError(
                "Extra edge attributes are unsupported by this wire format"
            )
        edges.append((key, describe(block.edge_index), block.num_sampled_edges))
    metadata = {
        "version": 1,
        "batch_size": batch_size or 0,
        "nodes": nodes,
        "edges": edges,
    }
    return msgpack.packb(metadata, use_bin_type=True), buffers


def unpack_python_hetero(metadata: bytes, buffers: list[TensorBuffer]) -> HeteroData:
    """Reconstruct CPU graph containers directly from version-1 NumPy wire data.

    Args:
        metadata: MessagePack emitted by the Python or Rust graph encoder.
        buffers: Contiguous numeric arrays whose owners are retained by Torch.
            Received read-only Ray arrays must not be modified through the tensors.

    Returns:
        HeteroData with zero-copy tensor views, without Arrow or Rust decoding.

    Raises:
        ValueError: If wire version, tensor layout, or graph structure is invalid.
        TypeError: If the payload contains unsupported buffer types.
    """
    import math

    import msgpack
    from kumoapi.typing import Stype
    from torch_frame import TensorFrame
    from torch_frame.data import MultiNestedTensor

    if not isinstance(metadata, bytes) or not isinstance(buffers, list):
        raise TypeError("Heterogeneous wire data requires bytes and a buffer list")
    spec = msgpack.unpackb(metadata, raw=False)
    if spec["version"] != 1:
        raise ValueError("Unsupported heterogeneous wire version")
    used: set[int] = set()
    supported_dtypes = (np.dtype("int64"), np.dtype("float32"), np.dtype("float64"))

    def restore(description: _TensorSpec) -> torch.Tensor:
        index = description["buffer"]
        if not isinstance(index, int) or not 0 <= index < len(buffers) or index in used:
            raise ValueError("Missing or repeated heterogeneous buffer index")
        array = buffers[index]
        if (
            not isinstance(array, np.ndarray)
            or array.ndim != 1
            or not array.flags.c_contiguous
            or not array.flags.aligned
            or not array.dtype.isnative
            or array.dtype not in supported_dtypes
        ):
            raise TypeError("Expected contiguous native-endian numeric buffers")
        shape = description["shape"]
        if (
            str(array.dtype) != description["dtype"]
            or not 1 <= len(shape) <= 4
            or any(not isinstance(size, int) or size < 0 for size in shape)
            or math.prod(shape) != array.size
        ):
            raise ValueError(
                "Heterogeneous buffer dtype or shape differs from metadata"
            )
        used.add(index)
        return torch.from_numpy(array).reshape(shape)

    data = HeteroData()
    for name, node in spec["nodes"].items():
        block = data[name]
        for key, description in node["attributes"].items():
            block[key] = restore(description)
        values, names = {}, {}
        rows = node["rows"]
        for feature in node["features"]:
            kind = Stype(feature["kind"])
            columns = feature["columns"]
            if kind in values or not columns:
                raise ValueError("Duplicate feature kind or empty feature columns")
            tensor = restore(feature["value"])
            offset_spec = feature["offset"]
            if kind.value == "multicategorical" and offset_spec is not None:
                offset = restore(offset_spec)
                offsets = offset.numpy()
                if (
                    tensor.dtype != torch.int64
                    or tensor.ndim != 1
                    or offset.dtype != torch.int64
                    or tuple(offset.shape) != (rows * len(columns) + 1,)
                    or offsets[0] != 0
                    or offsets[-1] != tensor.numel()
                    or np.any(offsets[1:] < offsets[:-1])
                ):
                    raise ValueError("Invalid ragged feature offsets")
                values[kind] = MultiNestedTensor(
                    num_rows=rows, num_cols=len(columns), values=tensor, offset=offset
                )
            elif (
                kind.value in ("numerical", "categorical", "timestamp")
                and offset_spec is None
            ):
                shape: tuple[int, ...] = (rows, len(columns))
                if kind.value == "timestamp":
                    shape += (16,)
                if tuple(tensor.shape) != shape:
                    raise ValueError("Feature shape differs from node rows/columns")
                values[kind] = tensor
            else:
                raise ValueError("Unsupported heterogeneous feature kind/layout")
            names[kind] = columns
        block.tf = TensorFrame(values, names, num_rows=rows)
        block.batch_size = spec["batch_size"]
        if node["empty_pkey"]:
            block.pkey = np.empty(0, dtype=object)
        if node["counts"] is not None:
            block.num_sampled_nodes = node["counts"]
    seen_edges: set[tuple[str, str, str]] = set()
    for key, description, counts in spec["edges"]:
        edge_key = tuple(key)
        tensor = restore(description)
        if (
            len(edge_key) != 3
            or edge_key in seen_edges
            or edge_key[0] not in spec["nodes"]
            or edge_key[2] not in spec["nodes"]
            or tensor.ndim != 2
            or tensor.shape[0] != 2
            or tensor.dtype != torch.int64
        ):
            raise ValueError("Invalid heterogeneous edge layout or node types")
        seen_edges.add(edge_key)
        data[edge_key].edge_index = tensor
        data[edge_key].num_sampled_edges = counts
    if len(used) != len(buffers):
        raise ValueError("Unreferenced heterogeneous buffers")
    return data
