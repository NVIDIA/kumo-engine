# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt explicit Kumo PyTorch graph outputs without importing an optional backend.

Use kumo.pytorch.api to prepare/materialize batches as in examples/kumo-engine.
This adapter accepts explicitly shared or disjoint native graph tensors; it does not assume
native node IDs are DataFrame offsets or infer historical availability clocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from tfm.contracts.graph import EdgeType, GraphAnchor, GraphBatch


class NativeGraphBatch(Protocol):
    """The public Kumo batch fields used by this adapter."""

    inputs: Mapping[str, Any]
    metadata: Mapping[str, Any]


@torch.inference_mode(False)
@torch.no_grad()
def from_kumo_graph(
    batch: NativeGraphBatch,
    *,
    anchors: tuple[GraphAnchor, ...],
    root_type: str,
    root_node_ids: tuple[int, ...],
    feature_columns: Mapping[str, tuple[str, ...]],
    relations: tuple[EdgeType, ...],
    forbidden_fields: frozenset[str],
    node_available_us: dict[str, Tensor],
    edge_available_us: dict[EdgeType, Tensor],
    graph_input: str = "graph",
    sharing: Literal["disjoint", "shared"] = "disjoint",
    node_hop: dict[str, Tensor] | None = None,
    num_sampled_nodes: dict[str, tuple[int, ...]] | None = None,
    num_sampled_edges: dict[EdgeType, tuple[int, ...]] | None = None,
    hop_depth: int | None = None,
) -> GraphBatch:
    """Map native graph IDs/roles to the experimental CPU baseline contract.

    Graph nodes expose `ids`, `batch`, and `features`; relation edges expose
    `index`, `source_type`, and `destination_type`. Raw/fitted numeric
    feature columns and availability tensors must align with that native output.
    The caller obtains canonical root IDs from its anchor binding. Unknown
    masks, crossed query graphs, missing/reordered samples, and duplicate roots
    fail before model execution. Supply node_hop and both count dictionaries for
    an already ordered native graph, or hop_depth to construct Engine-compatible
    prefixes on CPU. Omitting both preserves an untrimmed graph and cannot be
    consumed by an encoder with trim=True. No optional dependency is imported
    in core CI.
    """
    if tuple(batch.metadata["sample_ids"]) != tuple(a.sample_id for a in anchors):
        raise ValueError("Kumo sample order differs from requested anchors")
    if len(root_node_ids) != len(anchors) or root_type not in feature_columns:
        raise ValueError("Kumo adapter requires one bound root ID per anchor")
    graph = batch.inputs[graph_input]
    nodes = graph["nodes"]
    features, owners = {}, {}
    for name, columns in feature_columns.items():
        if forbidden_fields & set(columns):
            raise ValueError("Forbidden field in native graph feature allowlist")
        if not columns:
            raise ValueError("Declare fitted numeric graph feature columns explicitly")
        fields = nodes[name]["features"]
        blocks = []
        for column in columns:
            field = fields[column]
            if "valid" in field and not bool(field["valid"].all()):
                raise ValueError(
                    "Impute missing native features explicitly, including presence flags"
                )
            values = field["values"]
            if values.ndim != 1 or not values.is_floating_point():
                raise ValueError("Kumo graph features must be pre-encoded scalar floats")
            blocks.append(values)
        features[name] = torch.stack(blocks, dim=1)
        owners[name] = (
            nodes[name].get(
                "batch",
                torch.zeros(len(features[name]), dtype=torch.long, device=features[name].device),
            )
            if sharing == "shared"
            else nodes[name]["batch"]
        )
    roots = []
    for i, root in enumerate(root_node_ids):
        found = (
            (
                (nodes[root_type]["ids"] == root)
                & (owners[root_type] == (0 if sharing == "shared" else i))
            )
            .nonzero()
            .flatten()
        )
        if found.numel() != 1:
            raise ValueError("Missing or duplicate native root for anchor")
        roots.append(int(found[0]))
    edges: dict[EdgeType, Tensor] = {}
    clocks: dict[EdgeType, Tensor] = {}
    for source, role, target in relations:
        if role in graph["edges"]:
            native = graph["edges"][role]
            if (native["source_type"], native["destination_type"]) != (source, target):
                raise ValueError("Native edge orientation differs from declared FK role")
            edges[source, role, target] = native["index"]
            clocks[source, role, target] = edge_available_us[source, role, target]
        elif role.startswith("rev_") and role[4:] in graph["edges"]:
            # The graph schema defines rev_<role> mechanically. Kumo can emit
            # the physical relation once; preserve its ordering and timestamps
            # while supplying the explicitly declared reverse message direction.
            native = graph["edges"][role[4:]]
            if (native["source_type"], native["destination_type"]) != (target, source):
                raise ValueError("Native reverse edge orientation differs from declared FK role")
            edges[source, role, target] = native["index"].flip(0)
            clocks[source, role, target] = edge_available_us[target, role[4:], source]
        else:
            raise ValueError(f"Native graph lacks declared relation role {role!r}")
    root_index = torch.tensor(roots, dtype=torch.long, device=owners[root_type].device)
    result = GraphBatch(
        anchors,
        root_type,
        features,
        owners,
        node_available_us,
        edges,
        clocks,
        root_index,
        sharing=sharing,
        node_hop=node_hop,
        num_sampled_nodes=num_sampled_nodes,
        num_sampled_edges=num_sampled_edges,
    )
    if hop_depth is not None:
        if num_sampled_nodes is not None:
            raise ValueError("Supply native hop counts or request hop construction, not both")
        from tfm.data.graph import order_graph_by_hop

        return order_graph_by_hop(result, hop_depth)
    return result
